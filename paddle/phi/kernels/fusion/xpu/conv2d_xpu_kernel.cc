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
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {
namespace fusion {

template <typename T_X,
          typename T_W,
          typename T_OUT,
          typename T_GEMM,
          typename Context>
void Conv2dXPUKernelImpl(const Context& ctx,
                         const DenseTensor& x,
                         const paddle::optional<DenseTensor>& x_max,
                         const DenseTensor& filter,
                         const DenseTensor& filter_max,
                         const paddle::optional<DenseTensor>& bias,
                         const paddle::optional<DenseTensor>& branch,
                         const paddle::optional<DenseTensor>& branch_max,
                         const paddle::optional<DenseTensor>& scale_max,
                         const paddle::optional<DenseTensor>& out_max_in,
                         const std::vector<int>& paddings,
                         const std::vector<int>& dilations,
                         const std::vector<int>& strides,
                         const std::string& padding_algorithm,
                         int groups,
                         int act_type,
                         float act_param,
                         DenseTensor* out,
                         DenseTensor* out_max) {
  using XPUTypeX = typename XPUTypeTrait<T_X>::Type;
  using XPUTypeW = typename XPUTypeTrait<T_W>::Type;
  using XPUTypeOut = typename XPUTypeTrait<T_OUT>::Type;
  auto input_dims = x.dims();
  auto filter_dims = filter.dims();
  // update paddings and dilations according to padding_algorithm
  std::vector<int> paddings_vec = paddings;
  std::vector<int> dilations_vec = dilations;
  DDim in_data_dims = common::slice_ddim(input_dims, 2, input_dims.size());
  DDim filter_data_dims =
      common::slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  phi::UpdatePaddingAndDilation(&paddings_vec,
                                &dilations_vec,
                                padding_algorithm,
                                in_data_dims,
                                strides,
                                ksize);

  int batch = static_cast<int>(input_dims[0]);
  int in_c = static_cast<int>(input_dims[1]);
  int in_h = static_cast<int>(input_dims[2]);
  int in_w = static_cast<int>(input_dims[3]);
  int out_c = static_cast<int>(filter_dims[0]);
  int win_h = static_cast<int>(filter_dims[2]);
  int win_w = static_cast<int>(filter_dims[3]);
  auto* input_data = reinterpret_cast<const XPUTypeX*>(x.data<T_X>());
  const float* input_max_data =
      x_max.get_ptr() == nullptr ? nullptr : x_max.get_ptr()->data<float>();
  auto* filter_data = reinterpret_cast<const XPUTypeW*>(filter.data<T_W>());
  auto* filter_max_data = filter_max.data<float>();
  auto* scale_max_data = scale_max.get_ptr() == nullptr
                             ? nullptr
                             : scale_max.get_ptr()->data<float>();

  const XPUTypeOut* branch_data = nullptr;
  const float* branch_max_data = branch_max.get_ptr() == nullptr
                                     ? nullptr
                                     : branch_max.get_ptr()->data<float>();
  auto* branch_tensor = branch.get_ptr();
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  if (branch_tensor != nullptr) {
    if (branch_tensor->dtype() == out->dtype()) {
      branch_data =
          reinterpret_cast<const XPUTypeOut*>(branch_tensor->data<T_OUT>());
    } else {
      auto branch_data_temp =
          RAII_GUARD.alloc_l3_or_gm<XPUTypeOut>(branch_tensor->numel());
      int r = xpu::cast<XPUTypeX, XPUTypeOut>(
          ctx.x_context(),
          reinterpret_cast<const XPUTypeX*>(branch_tensor->data<T_X>()),
          branch_data_temp,
          branch_tensor->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
      branch_data = branch_data_temp;
    }
  }

  const float* bias_data =
      bias.get_ptr() == nullptr ? nullptr : bias.get_ptr()->data<float>();
  auto* out_data =
      reinterpret_cast<XPUTypeOut*>(ctx.template Alloc<T_OUT>(out));
  auto* out_max_data = ctx.template Alloc<float>(out_max);
  out_max_data = out_max_in.get_ptr() != nullptr
                     ? const_cast<float*>(out_max_in.get_ptr()->data<float>())
                     : out_max_data;
  xpu::Activation_t act(static_cast<xpu::Activation_t::act_enum>(act_type));
  if (act_type == xpu::Activation_t::LEAKY_RELU) {
    act.leaky_alpha = act_param;
  } else if (act_type == xpu::Activation_t::HARD_SIGMOID) {
    act.hard_sigmoid_slope = act_param;
  }

  int r = xpu::
      conv2d_fusion<XPUTypeX, XPUTypeW, XPUTypeOut, T_GEMM>(  // TX/TW/TY/TGEMM
          /* baidu::xpu::api::Context* ctx */ ctx.x_context(),
          /* const TX* input */ input_data,
          /* const TW* filter */ filter_data,
          /* TY* output */ out_data,
          /* int64_t n */ batch,
          /* int64_t ic */ in_c,
          /* int64_t h */ in_h,
          /* int64_t w */ in_w,
          /* int64_t oc */ out_c,
          /* const std::vector<int>& ksize */ std::vector<int>{win_h, win_w},
          /* const std::vector<int>& strides */ strides,
          /* const std::vector<int>& paddings */ paddings_vec,
          /* const std::vector<int>& dilations */ dilations_vec,
          /* int64_t groups */ groups,
          /* const float* in_maxptr */ input_max_data,
          /* const float* filter_maxptr */ filter_max_data,
          /* float* out_maxptr */ out_max_data,
          /* bool is_nchw */ true,
          /* const float* bias */ bias_data,
          /* const TY* branch */ branch_data,
          /* const baidu::xpu::api::Activation_t& act */ act,
          /* const float* branch_maxptr */ branch_max_data,
          /* const float* scale */ scale_max_data);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_xpu");
}

#define CONV2D_XPU_KERNEL_IMPL(x_dtype_, w_dtype_, out_dtype_, gemm_dtype_)  \
  Conv2dXPUKernelImpl<x_dtype_, w_dtype_, out_dtype_, gemm_dtype_, Context>( \
      ctx,                                                                   \
      x,                                                                     \
      x_max,                                                                 \
      filter,                                                                \
      filter_max,                                                            \
      bias,                                                                  \
      branch,                                                                \
      branch_max,                                                            \
      scale_max,                                                             \
      out_max_in,                                                            \
      paddings,                                                              \
      dilations,                                                             \
      strides,                                                               \
      padding_algorithm,                                                     \
      groups,                                                                \
      act_type,                                                              \
      act_param,                                                             \
      out,                                                                   \
      out_max);

template <typename T, typename Context>
void Conv2dXPUKernel(const Context& ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& x_max,
                     const DenseTensor& filter,
                     const DenseTensor& filter_max,
                     const paddle::optional<DenseTensor>& bias,
                     const paddle::optional<DenseTensor>& branch,
                     const paddle::optional<DenseTensor>& branch_max,
                     const paddle::optional<DenseTensor>& scale_max,
                     const paddle::optional<DenseTensor>& out_max_in,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const std::string& padding_algorithm,
                     int groups,
                     int act_type,
                     float act_param,
                     DataType out_dtype,
                     DenseTensor* out,
                     DenseTensor* out_max) {
  // Dont use template T param
  VLOG(4) << "Conv kernel type: " << x.dtype() << " ," << filter.dtype() << " ,"
          << out_dtype;
  if (x.dtype() == DataType::FLOAT32) {
    // float32/float16 kernel
    if (filter.dtype() == DataType::INT16) {
      if (out_dtype == DataType::FLOAT32) {
        CONV2D_XPU_KERNEL_IMPL(float, int16_t, float, int16_t);
      } else if (out_dtype == DataType::FLOAT16) {
        CONV2D_XPU_KERNEL_IMPL(float, int16_t, dtype::float16, int16_t);
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Not support x_dtype is %s, filter_dtype is %s and out_dtype is "
            "%s.",
            DataTypeToString(x.dtype()),
            DataTypeToString(filter.dtype()),
            DataTypeToString(out_dtype)));
      }
    } else if (filter.dtype() == DataType::INT8) {
      if (out_dtype == DataType::FLOAT32) {
        CONV2D_XPU_KERNEL_IMPL(float, int8_t, float, int8_t);
      } else if (out_dtype == DataType::INT8) {
        CONV2D_XPU_KERNEL_IMPL(float, int8_t, int8_t, int8_t);
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Not support x_dtype is %s, filter_dtype is %s and out_dtype is "
            "%s.",
            DataTypeToString(x.dtype()),
            DataTypeToString(filter.dtype()),
            DataTypeToString(out_dtype)));
      }
    } else if (filter.dtype() == DataType::FLOAT32) {
      CONV2D_XPU_KERNEL_IMPL(float, float, float, int32_t);
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Not support x_dtype is %s, filter_dtype is %s and out_dtype is %s.",
          DataTypeToString(x.dtype()),
          DataTypeToString(filter.dtype()),
          DataTypeToString(out_dtype)));
    }
    return;
  }

  if (x.dtype() == DataType::FLOAT16) {
    // float16 kernel
    if (filter.dtype() == DataType::INT16) {
      if (out_dtype == DataType::FLOAT32) {
        CONV2D_XPU_KERNEL_IMPL(phi::dtype::float16, int16_t, float, int16_t);
      } else if (out_dtype == DataType::FLOAT16) {
        CONV2D_XPU_KERNEL_IMPL(
            phi::dtype::float16, int16_t, dtype::float16, int16_t);
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Not support x_dtype is %s, filter_dtype is %s and out_dtype is "
            "%s.",
            DataTypeToString(x.dtype()),
            DataTypeToString(filter.dtype()),
            DataTypeToString(out_dtype)));
      }
    } else if (filter.dtype() == DataType::INT8) {
      if (out_dtype == DataType::FLOAT16) {
        CONV2D_XPU_KERNEL_IMPL(
            phi::dtype::float16, int8_t, dtype::float16, int8_t);
      } else if (out_dtype == DataType::INT8) {
        CONV2D_XPU_KERNEL_IMPL(phi::dtype::float16, int8_t, int8_t, int8_t);
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Not support x_dtype is %s, filter_dtype is %s and out_dtype is "
            "%s.",
            DataTypeToString(x.dtype()),
            DataTypeToString(filter.dtype()),
            DataTypeToString(out_dtype)));
      }
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Not support x_dtype is %s, filter_dtype is %s and out_dtype is %s.",
          DataTypeToString(x.dtype()),
          DataTypeToString(filter.dtype()),
          DataTypeToString(out_dtype)));
    }
    return;
  }

  if (x.dtype() == DataType::INT8) {
    if (filter.dtype() == DataType::INT8) {
      if (out_dtype == DataType::FLOAT32) {
        CONV2D_XPU_KERNEL_IMPL(int8_t, int8_t, float, int8_t);
      } else if (out_dtype == DataType::FLOAT16) {
        CONV2D_XPU_KERNEL_IMPL(int8_t, int8_t, dtype::float16, int8_t);
      } else if (out_dtype == DataType::INT8) {
        CONV2D_XPU_KERNEL_IMPL(int8_t, int8_t, int8_t, int8_t);
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "Not support x_dtype is %s, filter_dtype is %s and out_dtype is "
            "%s.",
            DataTypeToString(x.dtype()),
            DataTypeToString(filter.dtype()),
            DataTypeToString(out_dtype)));
      }
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Not support x_dtype is %s, filter_dtype is %s and out_dtype is %s.",
          DataTypeToString(x.dtype()),
          DataTypeToString(filter.dtype()),
          DataTypeToString(out_dtype)));
    }
    return;
  }

  PADDLE_THROW(common::errors::Unimplemented(
      "Not support x_dtype is %s, filter_dtype is %s and out_dtype is %s.",
      DataTypeToString(x.dtype()),
      DataTypeToString(filter.dtype()),
      DataTypeToString(out_dtype)));
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(conv2d_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::Conv2dXPUKernel,
                   float,
                   phi::dtype::float16,
                   int8_t) {}
