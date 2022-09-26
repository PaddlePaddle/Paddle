// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"

namespace paddle {
namespace distributed {

template <typename DeviceContext, typename T>
struct SplitDense2Dense {
  void operator()(const DeviceContext *context,
                  phi::DenseTensor *in,
                  std::vector<phi::DenseTensor *> *out,
                  int axis = 0) {
    std::vector<const phi::DenseTensor *> shape_refer;
    shape_refer.reserve(out->size());
    for (auto &p_tensor : *out) {
      shape_refer.emplace_back(p_tensor);
    }
    phi::funcs::SplitFunctor<DeviceContext, T> split_functor_;
    split_functor_(*context, *in, shape_refer, axis, out);
  }
};

template <typename DeviceContext>
void SplitDense2DenseWithType(const DeviceContext *dev_ctx,
                              phi::DenseTensor *p_dense,
                              std::vector<phi::DenseTensor *> *p_list,
                              phi::DataType type) {
  switch (type) {
    case phi::DataType::BOOL:
      SplitDense2Dense<DeviceContext, bool>()(dev_ctx, p_dense, p_list);
      break;
    case phi::DataType::UINT8:
      SplitDense2Dense<DeviceContext, uint8_t>()(dev_ctx, p_dense, p_list);
      break;
    case phi::DataType::INT8:
      SplitDense2Dense<DeviceContext, int8_t>()(dev_ctx, p_dense, p_list);
      break;
    case phi::DataType::INT32:
      SplitDense2Dense<DeviceContext, int32_t>()(dev_ctx, p_dense, p_list);
      break;
    case phi::DataType::INT64:
      SplitDense2Dense<DeviceContext, int64_t>()(dev_ctx, p_dense, p_list);
      break;
    case phi::DataType::FLOAT16:
      SplitDense2Dense<DeviceContext, platform::float16>()(
          dev_ctx, p_dense, p_list);
      break;
    case phi::DataType::FLOAT32:
      SplitDense2Dense<DeviceContext, float>()(dev_ctx, p_dense, p_list);
      break;
    case phi::DataType::FLOAT64:
      SplitDense2Dense<DeviceContext, double>()(dev_ctx, p_dense, p_list);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it splits tensors for "
          "allgather.",
          type));
  }
}

void SplitDense2Tensor(
    phi::DenseTensor &tensor,                        // NOLINT
    std::vector<experimental::Tensor> &tensor_list,  // NOLINT
    const phi::DeviceContext *dev_ctx) {
  std::vector<phi::DenseTensor *> dense_list;
  for (auto &tensor : tensor_list) {
    auto p_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl()).get();
    dense_list.emplace_back(p_tensor);
  }

  const auto &place = dev_ctx->GetPlace();
  if (platform::is_gpu_place(place)) {
    SplitDense2DenseWithType(static_cast<const phi::GPUContext *>(dev_ctx),
                             &tensor,
                             &dense_list,
                             tensor.dtype());
  } else if (platform::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    SplitDense2DenseWithType(
        static_cast<const platform::CustomDeviceContext *>(dev_ctx),
        &tensor,
        &dense_list,
        tensor.dtype());
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't split tensor since it's not compiled with CUSTOM_DEVICE,"
        "Please recompile or reinstall Paddle with CUSTOM_DEVICE support."));
#endif
  } else if (platform::is_cpu_place(place)) {
    SplitDense2DenseWithType(static_cast<const phi::CPUContext *>(dev_ctx),
                             &tensor,
                             &dense_list,
                             tensor.dtype());
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Split tensor not supported on place (%s)", place));
  }
}

}  //  namespace distributed
}  //  namespace paddle
