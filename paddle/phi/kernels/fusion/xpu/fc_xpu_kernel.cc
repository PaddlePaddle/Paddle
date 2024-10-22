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
#ifdef PADDLE_WITH_XPU_XHPC
#include "xblas/cublasLt.h"
#endif

namespace phi {
namespace fusion {

template <typename T_X,
          typename T_W,
          typename T_OUT,
          typename T_GEMM,
          typename Context>
void FcXPUKernelImpl(const Context& ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& x_max,
                     const DenseTensor& w,
                     const DenseTensor& w_max,
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
  auto* w_max_data = w_max.data<float>();
  const float* bias_data =
      bias.get_ptr() == nullptr ? nullptr : bias.get_ptr()->data<float>();
  auto* out_data =
      reinterpret_cast<XPUTypeOut*>(ctx.template Alloc<T_OUT>(out));

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
  int r = 0;

#ifdef PADDLE_WITH_XPU_XHPC
  if (std::is_same<XPUTypeX, XPUTypeOut>::value) {
    r = baidu::xpu::xblas::
        fc_fusion<XPUTypeX, XPUTypeW, XPUTypeOut, T_GEMM>(  // TX, TW. TY TGEMM
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
            nullptr,                                        // scale_x
            nullptr,                                        // scale_w
            0,                                              // scale_x_mode
            0);                                             // scale_w_mode
  } else {
    auto* scale_max_data = scale_max.get_ptr() == nullptr
                               ? nullptr
                               : scale_max.get_ptr()->data<float>();
    r = xpu::
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
            act,
            scale_max_data);
  }
#else
  auto* scale_max_data = scale_max.get_ptr() == nullptr
                             ? nullptr
                             : scale_max.get_ptr()->data<float>();
  r = xpu::fc_fusion<XPUTypeX, XPUTypeW, XPUTypeOut, T_GEMM>(  // TX/TW/TY/TGEMM
      ctx.x_context(),                                         // ctx
      x_data,                                                  // x
      w_data,                                                  // w
      out_data,                                                // y
      m,                                                       // m
      n,                                                       // n
      k,                                                       // k
      transpose_x,                                             // x_trans
      true,                                                    // w_trans
      x_max_data,                                              // x_maxptr
      w_max_data,                                              // w_maxptr
      out_max_data,                                            // y_maxptr
      transpose_x ? m : k,                                     // ldx
      k,                                                       // ldw
      n,                                                       // ldy
      alpha,                                                   // alpha
      beta,                                                    // beta
      bias_data,                                               // bias
      act,
      scale_max_data);
#endif

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_xpu");
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
                 const DenseTensor& w_max,
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
                   int8_t) {}
