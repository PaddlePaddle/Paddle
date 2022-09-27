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

#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/device_manager.h"

namespace paddle {
namespace distributed {

template <typename DeviceContext, typename T>
struct SplitDenseTensor {
  void operator()(const DeviceContext *context,
                  const phi::DenseTensor &in,
                  std::vector<phi::DenseTensor *> *out,
                  int axis = 0) {
    std::vector<const phi::DenseTensor *> shape_refer;
    shape_refer.reserve(out->size());
    for (auto *p_tensor : *out) {
      shape_refer.emplace_back(p_tensor);
    }
    operators::math::SplitFunctor<DeviceContext, T> split_functor_;
    split_functor_(*context, in, shape_refer, axis, out);
  }
};

#ifdef PADDLE_WITH_CUSTOM_DEVICE
template <typename T>
struct SplitDenseTensor<platform::CustomDeviceContext, T> {
  void operator()(const platform::CustomDeviceContext *context,
                  const phi::DenseTensor &in,
                  std::vector<phi::DenseTensor *> *out) {
    auto *in_data = in.data<T>();
    auto *device = phi::DeviceManager::GetDeviceWithPlace(context->GetPlace());
    size_t offset = 0;
    for (auto *p_tensor : *out) {
      auto *out_data = p_tensor->data<T>();
      auto sz = p_tensor->numel() * sizeof(T);
      device->MemoryCopyD2D(out_data, in_data + offset, sz, nullptr);
      offset += sz;
    }
  }
};
#endif

template <typename DeviceContext>
void SplitDenseTensorWithType(const DeviceContext *dev_ctx,
                              const phi::DenseTensor &p_dense,
                              std::vector<phi::DenseTensor *> *p_list,
                              phi::DataType type) {
  switch (type) {
    case phi::DataType::BOOL:
      SplitDenseTensor<DeviceContext, bool>()(dev_ctx, p_dense, p_list);
      break;
    case phi::DataType::UINT8:
      SplitDenseTensor<DeviceContext, uint8_t>()(dev_ctx, p_dense, p_list);
      break;
    case phi::DataType::INT8:
      SplitDenseTensor<DeviceContext, int8_t>()(dev_ctx, p_dense, p_list);
      break;
    case phi::DataType::INT32:
      SplitDenseTensor<DeviceContext, int32_t>()(dev_ctx, p_dense, p_list);
      break;
    case phi::DataType::INT64:
      SplitDenseTensor<DeviceContext, int64_t>()(dev_ctx, p_dense, p_list);
      break;
    case phi::DataType::FLOAT16:
      SplitDenseTensor<DeviceContext, platform::float16>()(
          dev_ctx, p_dense, p_list);
      break;
    case phi::DataType::FLOAT32:
      SplitDenseTensor<DeviceContext, float>()(dev_ctx, p_dense, p_list);
      break;
    case phi::DataType::FLOAT64:
      SplitDenseTensor<DeviceContext, double>()(dev_ctx, p_dense, p_list);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it splits tensors for "
          "allgather.",
          type));
  }
}

void SplitTensor(const phi::DeviceContext *dev_ctx,
                 const phi::DenseTensor &tensor,
                 const std::vector<experimental::Tensor> *tensor_list) {
  std::vector<phi::DenseTensor *> dense_list;
  for (auto &tensor : *tensor_list) {
    auto p_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl()).get();
    dense_list.emplace_back(p_tensor);
  }

  const auto &place = dev_ctx->GetPlace();
  if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_GPU
    SplitDenseTensorWithType(static_cast<const phi::GPUContext *>(dev_ctx),
                             tensor,
                             &dense_list,
                             tensor.dtype());
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't split tensor since it's not support NCCL/RCCL, please "
        "recompile or reinstall Paddle with NCCL/RCCL support."));
#endif
  } else if (platform::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    SplitDenseTensorWithType(
        static_cast<const platform::CustomDeviceContext *>(dev_ctx),
        tensor,
        &dense_list,
        tensor.dtype());
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't split tensor since it's not compiled with CUSTOM_DEVICE, "
        "please recompile or reinstall Paddle with CUSTOM_DEVICE support."));
#endif
  } else if (platform::is_cpu_place(place)) {
    SplitDenseTensorWithType(static_cast<const phi::CPUContext *>(dev_ctx),
                             tensor,
                             &dense_list,
                             tensor.dtype());
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Split tensor not supported on place (%s)", place));
  }
}

}  //  namespace distributed
}  //  namespace paddle
