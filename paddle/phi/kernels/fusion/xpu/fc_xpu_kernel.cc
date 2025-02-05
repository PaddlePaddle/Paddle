// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "glog/logging.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

#ifdef PADDLE_WITH_XPU_XRE5
#include "xblas/cublasLt.h"
namespace xblas = baidu::xpu::xblas;
#endif

namespace phi {
namespace fusion {

using XPUTypeFP16 = typename XPUTypeTrait<phi::dtype::float16>::Type;
using XPUTypeBF16 = typename XPUTypeTrait<phi::dtype::bfloat16>::Type;

template <typename T_X,
          typename T_W,
          typename T_OUT,
          typename T_GEMM,
          typename Context>
void FcXPUKernelImpl(const Context& ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& x_max,
                     const DenseTensor& w,
                     const paddle::optional<DenseTensor>& w_max,
                     const paddle::optional<DenseTensor>& bias,
                     const paddle::optional<DenseTensor>& scale_max,
                     const paddle::optional<DenseTensor>& out_max_in,
                     int in_num_col_dims,
                     bool transpose_x,
                     float alpha,
                     float beta,
                     int act_type,
                     float act_alpha,
                     DenseTensor* out,
                     DenseTensor* out_max) {
  using XPUTypeX = typename XPUTypeTrait<T_X>::Type;
  using XPUTypeW = typename XPUTypeTrait<T_W>::Type;
  using XPUTypeOut = typename XPUTypeTrait<T_OUT>::Type;
  auto in_mat_dims = flatten_to_2d(x.dims(), in_num_col_dims);
  int m = in_mat_dims[0];
  int k = in_mat_dims[1];
  int n = w.dims()[0];
  auto* x_data = reinterpret_cast<const XPUTypeX*>(x.data<T_X>());
  const float* x_max_data =
      x_max.get_ptr() == nullptr ? nullptr : x_max.get_ptr()->data<float>();
  auto* w_data = reinterpret_cast<const XPUTypeW*>(w.data<T_W>());
  const float* w_max_data =
      w_max.get_ptr() == nullptr ? nullptr : w_max.get_ptr()->data<float>();
  const float* bias_data =
      bias.get_ptr() == nullptr ? nullptr : bias.get_ptr()->data<float>();
  auto* out_data =
      reinterpret_cast<XPUTypeOut*>(ctx.template Alloc<T_OUT>(out));
  auto* scale_max_data = scale_max.get_ptr() == nullptr
                             ? nullptr
                             : scale_max.get_ptr()->data<float>();
  float* out_max_data = nullptr;
  // when T_OUT is float and TGEMM is int8_t, out_max_data should better set to
  // nullptr for better performance
  if (!(std::is_same<T_OUT, float>::value &&
        std::is_same<T_GEMM, int8_t>::value)) {
    out_max_data = ctx.template Alloc<float>(out_max);
    out_max_data = out_max_in.get_ptr() != nullptr
                       ? const_cast<float*>(out_max_in.get_ptr()->data<float>())
                       : out_max_data;
  }

  xpu::Activation_t act(static_cast<xpu::Activation_t::act_enum>(act_type));
  if (act_type == xpu::Activation_t::LEAKY_RELU) {
    act.leaky_alpha = act_alpha;
  } else if (act_type == xpu::Activation_t::HARD_SIGMOID) {
    act.hard_sigmoid_slope = act_alpha;
  }
  // only for xpu3
#ifdef PADDLE_WITH_XPU_XRE5
  if constexpr (std::is_same<T_X, bfloat16>::value &&
                std::is_same<T_W, bfloat16>::value &&
                std::is_same<T_OUT, bfloat16>::value) {
    // use xte to speedup bfloat16 calc
    // whether to enable this feature requires a trade-off between performance
    // precision
    if (std::getenv("XPU_PADDLE_FC_BFLOAT16_XTE") != nullptr) {
      xpu::ctx_guard RAII_GUARD(ctx.x_context());
      const int MAXPTR_N = ctx.x_context()->max_ptr_size();
      int x_len = m * k;
      XPUTypeFP16* x_data_fp16 = nullptr;
      x_data_fp16 = RAII_GUARD.alloc_l3_or_gm<XPUTypeFP16>(x_len);
      PADDLE_ENFORCE_XDNN_NOT_NULL(x_data_fp16);
      int w_len = k * n;
      XPUTypeFP16* w_data_fp16 = nullptr;
      w_data_fp16 = RAII_GUARD.alloc_l3_or_gm<XPUTypeFP16>(w_len);
      PADDLE_ENFORCE_XDNN_NOT_NULL(w_data_fp16);

      float* xte_scale_x = nullptr;
      float* xte_scale_w = nullptr;
      xte_scale_x = RAII_GUARD.alloc_l3_or_gm<float>(1);
      PADDLE_ENFORCE_XDNN_NOT_NULL(xte_scale_x);
      xte_scale_w = RAII_GUARD.alloc_l3_or_gm<float>(1);
      PADDLE_ENFORCE_XDNN_NOT_NULL(xte_scale_w);

      float* xte_x_maxptr = nullptr;
      float* xte_w_maxptr = nullptr;
      if (x_max_data == nullptr) {
        xte_x_maxptr = RAII_GUARD.alloc_l3_or_gm<float>(MAXPTR_N);
        PADDLE_ENFORCE_XDNN_NOT_NULL(xte_x_maxptr);
        int r = xpu::findmax(ctx.x_context(), x_data, xte_x_maxptr, x_len);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu_findmax");
        r = xpu::cast_te(ctx.x_context(),
                         x_data,
                         xte_x_maxptr,
                         x_data_fp16,
                         xte_scale_x,
                         x_len);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu_cast_te");
      } else {
        int r = xpu::cast_te(ctx.x_context(),
                             x_data,
                             x_max_data,
                             x_data_fp16,
                             xte_scale_x,
                             x_len);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu_cast_te");
      }
      if (w_max_data == nullptr) {
        xte_w_maxptr = RAII_GUARD.alloc_l3_or_gm<float>(MAXPTR_N);
        PADDLE_ENFORCE_XDNN_NOT_NULL(xte_w_maxptr);
        int r = xpu::findmax(ctx.x_context(), w_data, xte_w_maxptr, w_len);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu_findmax");
        r = xpu::cast_te(ctx.x_context(),
                         w_data,
                         xte_w_maxptr,
                         w_data_fp16,
                         xte_scale_w,
                         w_len);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu_cast_te");
      } else {
        int r = xpu::cast_te(ctx.x_context(),
                             w_data,
                             w_max_data,
                             w_data_fp16,
                             xte_scale_w,
                             w_len);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu_cast_te");
      }
      baidu::xpu::xblas::FcFusionTensor<const XPUTypeFP16> tensor_a1{
          x_data_fp16,
          x_max_data ? x_max_data : xte_x_maxptr,
          transpose_x ? k : m,
          transpose_x ? m : k,
          transpose_x ? m : k,
          transpose_x};
      baidu::xpu::xblas::FcFusionTensor<const XPUTypeFP16> tensor_b1{
          w_data_fp16, w_max_data ? w_max_data : xte_w_maxptr, n, k, k, true};
      baidu::xpu::xblas::FcFusionTensor<const XPUTypeBF16> tensor_c1{
          out_data, nullptr, m, n, n, false};
      baidu::xpu::xblas::FcFusionTensor<XPUTypeBF16> tensor_d1{
          out_data, nullptr, m, n, n, false};
      baidu::xpu::xblas::FcFusionDesc<XPUTypeFP16, float, float> desc{alpha,
                                                                      beta};

      baidu::xpu::xblas::FcFusionEpilogue<float, float> epilogue1{
          act, bias_data, xte_scale_x, xte_scale_w, 0, 0, out_max_data};

      if (act_type == xpu::Activation_t::SWISH_GLU) {
        tensor_d1 = baidu::xpu::xblas::FcFusionTensor<XPUTypeBF16>{
            out_data, nullptr, m, n / 2, n / 2, false};
      } else {
        tensor_d1 = baidu::xpu::xblas::FcFusionTensor<XPUTypeBF16>{
            out_data, nullptr, m, n, n, false};
      }

      int r = baidu::xpu::xblas::fc_fusion<XPUTypeFP16,
                                           XPUTypeFP16,
                                           XPUTypeBF16,
                                           XPUTypeBF16,
                                           XPUTypeFP16,
                                           float,
                                           float,
                                           float,
                                           float>(ctx.x_context(),
                                                  tensor_a1,
                                                  tensor_b1,
                                                  tensor_c1,
                                                  tensor_d1,
                                                  desc,
                                                  epilogue1);

      PADDLE_ENFORCE_XDNN_SUCCESS(r, "xblas_fc_fusion");
    }
  }
  if (std::getenv("XPU_PADDLE_FC_BFLOAT16_XTE") == nullptr) {
    if constexpr (((std::is_same<T_X, float16>::value &&
                    std::is_same<T_W, int16_t>::value &&
                    std::is_same<T_GEMM, int16_t>::value &&
                    std::is_same<T_OUT, float>::value) ||
                   (std::is_same<T_X, float>::value &&
                    std::is_same<T_W, int16_t>::value &&
                    std::is_same<T_GEMM, int16_t>::value &&
                    std::is_same<T_OUT, float16>::value) ||
                   (std::is_same<T_X, float>::value &&
                    std::is_same<T_W, signed char>::value &&
                    std::is_same<T_GEMM, signed char>::value &&
                    std::is_same<T_OUT, signed char>::value))) {
      int r = xpu::
          fc_fusion<XPUTypeX, XPUTypeW, XPUTypeOut, T_GEMM>(  // TX/TW/TY/TGEMM
              ctx.x_context(),                                // ctx
              x_data,                                         // x
              w_data,                                         // w
              out_data,                                       // y
              m,                                              // m
              n,                                              // n
              k,                                              // k
              transpose_x,                                    // x_trans
              true,                                           // w_trans
              x_max_data,                                     // x_maxptr
              w_max_data,                                     // w_maxptr
              out_max_data,                                   // y_maxptr
              transpose_x ? m : k,                            // ldx
              k,                                              // ldw
              n,                                              // ldy
              alpha,                                          // alpha
              beta,                                           // beta
              bias_data,                                      // bias
              act,                                            // act
              scale_max_data);                                // scale
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_xpu");
    } else {
      baidu::xpu::xblas::FcFusionTensor<const XPUTypeX> tensor_a1{
          x_data,
          x_max_data,
          transpose_x ? k : m,
          transpose_x ? m : k,
          transpose_x ? m : k,
          transpose_x};
      baidu::xpu::xblas::FcFusionTensor<const XPUTypeW> tensor_b1{
          w_data, w_max_data, n, k, k, true};
      baidu::xpu::xblas::FcFusionTensor<const XPUTypeOut> tensor_c1{
          out_data, nullptr, m, n, n, false};
      baidu::xpu::xblas::FcFusionTensor<XPUTypeOut> tensor_d1{
          out_data, nullptr, m, n, n, false};
      baidu::xpu::xblas::FcFusionDesc<T_GEMM, float, XPUTypeOut> desc{alpha,
                                                                      beta};

      baidu::xpu::xblas::FcFusionEpilogue<float, float> epilogue1{
          act, bias_data, scale_max_data, nullptr, 0, 0, out_max_data};

      if (act_type == xpu::Activation_t::SWISH_GLU) {
        tensor_d1 = baidu::xpu::xblas::FcFusionTensor<XPUTypeOut>{
            out_data, nullptr, m, n / 2, n / 2, false};
      } else {
        tensor_d1 = baidu::xpu::xblas::FcFusionTensor<XPUTypeOut>{
            out_data, nullptr, m, n, n, false};
      }
      int r = baidu::xpu::xblas::fc_fusion<XPUTypeX,
                                           XPUTypeW,
                                           XPUTypeOut,
                                           XPUTypeOut,
                                           T_GEMM,
                                           float,
                                           XPUTypeOut,
                                           float,
                                           float>(ctx.x_context(),
                                                  tensor_a1,
                                                  tensor_b1,
                                                  tensor_c1,
                                                  tensor_d1,
                                                  desc,
                                                  epilogue1);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "xblas_fc_fusion");
    }
  }
#else
  int r =
      xpu::fc_fusion<XPUTypeX, XPUTypeW, XPUTypeOut, T_GEMM>(  // TX/TW/TY/TGEMM
          ctx.x_context(),                                     // ctx
          x_data,                                              // x
          w_data,                                              // w
          out_data,                                            // y
          m,                                                   // m
          n,                                                   // n
          k,                                                   // k
          transpose_x,                                         // x_trans
          true,                                                // w_trans
          x_max_data,                                          // x_maxptr
          w_max_data,                                          // w_maxptr
          out_max_data,                                        // y_maxptr
          transpose_x ? m : k,                                 // ldx
          k,                                                   // ldw
          n,                                                   // ldy
          alpha,                                               // alpha
          beta,                                                // beta
          bias_data,                                           // bias
          act,                                                 // act
          scale_max_data);                                     // scale

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_xpu");
#endif
}

#define FC_XPU_KERNEL_IMPL(x_dtype_, w_dtype_, out_dtype_, gemm_dtype_) \
  FcXPUKernelImpl<x_dtype_, w_dtype_, out_dtype_, gemm_dtype_>(         \
      ctx,                                                              \
      x,                                                                \
      x_max,                                                            \
      w,                                                                \
      w_max,                                                            \
      bias,                                                             \
      scale_max,                                                        \
      out_max_in,                                                       \
      in_num_col_dims,                                                  \
      transpose_x,                                                      \
      alpha,                                                            \
      beta,                                                             \
      act_type,                                                         \
      act_alpha,                                                        \
      out,                                                              \
      out_max);

template <typename T, typename Context>
void FcXPUKernel(const Context& ctx,
                 const DenseTensor& x,
                 const paddle::optional<DenseTensor>& x_max,
                 const DenseTensor& w,
                 const paddle::optional<DenseTensor>& w_max,
                 const paddle::optional<DenseTensor>& bias,
                 const paddle::optional<DenseTensor>& scale_max,
                 const paddle::optional<DenseTensor>& out_max_in,
                 int in_num_col_dims,
                 bool transpose_x,
                 float alpha,
                 float beta,
                 int act_type,
                 float act_alpha,
                 DataType out_dtype,
                 DenseTensor* out,
                 DenseTensor* out_max) {
  // Dont use template T param
  VLOG(4) << "Fc kernel type: " << x.dtype() << " ," << w.dtype() << " ,"
          << out_dtype;
  if (x.dtype() == DataType::FLOAT32) {
    // float32/float16 kernel
    if (w.dtype() == DataType::INT16) {
      if (out_dtype == DataType::FLOAT32) {
        FC_XPU_KERNEL_IMPL(float, int16_t, float, int16_t);
      } else if (out_dtype == DataType::FLOAT16) {
        FC_XPU_KERNEL_IMPL(float, int16_t, dtype::float16, int16_t);
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Not support x_dtype is %s, w_dtype is %s and out_dtype is "
            "%s.",
            DataTypeToString(x.dtype()),
            DataTypeToString(w.dtype()),
            DataTypeToString(out_dtype)));
      }
    } else if (w.dtype() == DataType::INT8) {
      if (out_dtype == DataType::FLOAT32) {
        FC_XPU_KERNEL_IMPL(float, int8_t, float, int8_t);
      } else if (out_dtype == DataType::INT8) {
        FC_XPU_KERNEL_IMPL(float, int8_t, int8_t, int8_t);
      } else if (out_dtype == DataType::FLOAT16) {
        FC_XPU_KERNEL_IMPL(float, int8_t, dtype::float16, int8_t);
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Not support x_dtype is %s, w_dtype is %s and out_dtype is "
            "%s.",
            DataTypeToString(x.dtype()),
            DataTypeToString(w.dtype()),
            DataTypeToString(out_dtype)));
      }
    } else if (w.dtype() == DataType::FLOAT32) {
      FC_XPU_KERNEL_IMPL(float, float, float, int32_t);
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Not support x_dtype is %s, w_dtype is %s and out_dtype is %s.",
          DataTypeToString(x.dtype()),
          DataTypeToString(w.dtype()),
          DataTypeToString(out_dtype)));
    }
    return;
  }

  if (x.dtype() == DataType::FLOAT16) {
    // float16 kernel
    if (w.dtype() == DataType::INT16) {
      if (out_dtype == DataType::FLOAT32) {
        FC_XPU_KERNEL_IMPL(phi::dtype::float16, int16_t, float, int16_t);
      } else if (out_dtype == DataType::FLOAT16) {
        FC_XPU_KERNEL_IMPL(
            phi::dtype::float16, int16_t, dtype::float16, int16_t);
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Not support x_dtype is %s, w_dtype is %s and out_dtype is "
            "%s.",
            DataTypeToString(x.dtype()),
            DataTypeToString(w.dtype()),
            DataTypeToString(out_dtype)));
      }
    } else if (w.dtype() == DataType::INT8) {
      if (out_dtype == DataType::FLOAT16) {
        FC_XPU_KERNEL_IMPL(phi::dtype::float16, int8_t, dtype::float16, int8_t);
      } else if (out_dtype == DataType::INT8) {
        FC_XPU_KERNEL_IMPL(phi::dtype::float16, int8_t, int8_t, int8_t);
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Not support x_dtype is %s, w_dtype is %s and out_dtype is "
            "%s.",
            DataTypeToString(x.dtype()),
            DataTypeToString(w.dtype()),
            DataTypeToString(out_dtype)));
      }
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Not support x_dtype is %s, w_dtype is %s and out_dtype is %s.",
          DataTypeToString(x.dtype()),
          DataTypeToString(w.dtype()),
          DataTypeToString(out_dtype)));
    }
    return;
  }

  if (x.dtype() == DataType::INT8) {
    if (w.dtype() == DataType::INT8) {
      if (out_dtype == DataType::FLOAT32) {
        FC_XPU_KERNEL_IMPL(int8_t, int8_t, float, int8_t);
      } else if (out_dtype == DataType::FLOAT16) {
        FC_XPU_KERNEL_IMPL(int8_t, int8_t, dtype::float16, int8_t);
      } else if (out_dtype == DataType::INT8) {
        FC_XPU_KERNEL_IMPL(int8_t, int8_t, int8_t, int8_t);
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Not support x_dtype is %s, w_dtype is %s and out_dtype is "
            "%s.",
            DataTypeToString(x.dtype()),
            DataTypeToString(w.dtype()),
            DataTypeToString(out_dtype)));
      }
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Not support x_dtype is %s, w_dtype is %s and out_dtype is %s.",
          DataTypeToString(x.dtype()),
          DataTypeToString(w.dtype()),
          DataTypeToString(out_dtype)));
    }
    return;
  }

  if (x.dtype() == DataType::BFLOAT16) {
    // bfloat16 kernel
    if (w.dtype() == DataType::BFLOAT16) {
      if (out_dtype == DataType::BFLOAT16) {
        FC_XPU_KERNEL_IMPL(phi::dtype::bfloat16,
                           phi::dtype::bfloat16,
                           phi::dtype::bfloat16,
                           float);
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Not support x_dtype is %s, w_dtype is %s and out_dtype is "
            "%s.",
            DataTypeToString(x.dtype()),
            DataTypeToString(w.dtype()),
            DataTypeToString(out_dtype)));
      }
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Not support x_dtype is %s, w_dtype is %s and out_dtype is %s.",
          DataTypeToString(x.dtype()),
          DataTypeToString(w.dtype()),
          DataTypeToString(out_dtype)));
    }
    return;
  }

  PADDLE_THROW(common::errors::Unimplemented(
      "Not support x_dtype is %s, w_dtype is %s and out_dtype is %s.",
      DataTypeToString(x.dtype()),
      DataTypeToString(w.dtype()),
      DataTypeToString(out_dtype)));
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fc_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FcXPUKernel,
                   float,
                   phi::dtype::float16,
                   int8_t,
                   phi::dtype::bfloat16) {}
