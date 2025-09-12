// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/process_group_kernel_utils.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/device_context.h"

namespace paddle {
namespace distributed {

template <>
void SplitDenseTensorByNumelWithType(const phi::XPUContext &dev_ctx,
                                     const phi::DenseTensor &t_in,
                                     std::vector<phi::DenseTensor> *t_list,
                                     phi::DataType type) {
  switch (type) {
    case phi::DataType::FLOAT16:
      SplitDenseTensorByNumel<phi::XPUContext, phi::dtype::float16>()(
          dev_ctx, t_in, t_list);
      break;
    case phi::DataType::BFLOAT16:
      SplitDenseTensorByNumel<phi::XPUContext, phi::dtype::bfloat16>()(
          dev_ctx, t_in, t_list);
      break;
    case phi::DataType::FLOAT32:
      SplitDenseTensorByNumel<phi::XPUContext, float>()(dev_ctx, t_in, t_list);
      break;
    case phi::DataType::INT32:
      SplitDenseTensorByNumel<phi::XPUContext, int32_t>()(
          dev_ctx, t_in, t_list);
      break;
    case phi::DataType::INT64:
      SplitDenseTensorByNumel<phi::XPUContext, int64_t>()(
          dev_ctx, t_in, t_list);
      break;
    case phi::DataType::UINT8:
      SplitDenseTensorByNumel<phi::XPUContext, uint8_t>()(
          dev_ctx, t_in, t_list);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Data type (%s) is not supported when it splits tensors.", type));
  }
}

template <>
void ConcatDenseTensorByNumelWithType(
    const phi::XPUContext &dev_ctx,
    const std::vector<phi::DenseTensor> &t_list,
    phi::DenseTensor *p_out,
    phi::DataType type) {
  switch (type) {
    case phi::DataType::FLOAT16:
      ConcatDenseTensorByNumel<phi::XPUContext, phi::dtype::float16>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::BFLOAT16:
      ConcatDenseTensorByNumel<phi::XPUContext, phi::dtype::bfloat16>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::FLOAT32:
      ConcatDenseTensorByNumel<phi::XPUContext, float>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT32:
      ConcatDenseTensorByNumel<phi::XPUContext, int32_t>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT64:
      ConcatDenseTensorByNumel<phi::XPUContext, int64_t>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::UINT8:
      ConcatDenseTensorByNumel<phi::XPUContext, uint8_t>()(
          dev_ctx, t_list, p_out);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Data type (%s) is not supported when it concats tensors.", type));
  }
}

void ConcatTensorByNumel(const phi::DeviceContext &dev_ctx,
                         const std::vector<phi::DenseTensor> &tensor_list,
                         phi::DenseTensor *tensor) {
  const auto &place = dev_ctx.GetPlace();
  if (phi::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    ConcatDenseTensorByNumelWithType(
        static_cast<const phi::XPUContext &>(dev_ctx),
        tensor_list,
        tensor,
        tensor->dtype());
#else
    PADDLE_THROW(common::errors::PermissionDenied(
        "Paddle can't concat tensor since it's not support XPU, please "
        "recompile or reinstall Paddle with XPU support."));
#endif
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Concat tensor by numel not supported on place (%s)", place));
  }
}

void SplitTensorByNumel(const phi::DeviceContext &dev_ctx,
                        const phi::DenseTensor &tensor,
                        std::vector<phi::DenseTensor> *tensor_list) {
  const auto &place = dev_ctx.GetPlace();
  if (phi::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    SplitDenseTensorByNumelWithType(
        static_cast<const phi::XPUContext &>(dev_ctx),
        tensor,
        tensor_list,
        tensor.dtype());
#else
    PADDLE_THROW(common::errors::PermissionDenied(
        "Paddle can't split tensor since it's not compiled with XPU, "
        "please recompile or reinstall Paddle with XPU support."));
#endif
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Split tensor by numel not supported on place (%s)", place));
  }
}

}  // namespace distributed
}  // namespace paddle
