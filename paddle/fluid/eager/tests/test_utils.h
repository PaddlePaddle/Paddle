// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/top/api/include/tensor.h"

#include "paddle/top/core/dense_tensor.h"
#include "paddle/top/core/tensor_meta.h"

#include "paddle/fluid/eager/function_api.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"

namespace egr {

template <typename T>
bool CompareGradTensorWithValue(const pt::Tensor& target, T value) {
  egr::AutogradMeta* meta = egr::EagerUtils::unsafe_autograd_meta(target);
  auto grad_dense =
      std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
  T* ptr = grad_dense->mutable_data<T>();

  std::vector<T> host_data(grad_dense->numel());
  if (grad_dense->backend() == pt::Backend::kCUDA) {
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx = dynamic_cast<paddle::platform::CUDADeviceContext*>(
        pool.Get(paddle::platform::CUDAPlace()));
    auto stream = dev_ctx->stream();

    paddle::memory::Copy(paddle::platform::CPUPlace(), host_data.data(),
                         paddle::platform::CUDAPlace(), ptr,
                         sizeof(T) * grad_dense->numel(), stream);
    ptr = host_data.data();
  }

  for (int i = 0; i < grad_dense->numel(); i++) {
    if (ptr[i] != value) return false;
  }
  return true;
}

template <typename T>
bool CompareTensorWithValue(const pt::Tensor& target, T value) {
  auto dense_t = std::dynamic_pointer_cast<pt::DenseTensor>(target.impl());
  T* ptr = dense_t->mutable_data<T>();

  std::vector<T> host_data(dense_t->numel());
  if (dense_t->backend() == pt::Backend::kCUDA) {
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx = dynamic_cast<paddle::platform::CUDADeviceContext*>(
        pool.Get(paddle::platform::CUDAPlace()));
    auto stream = dev_ctx->stream();

    paddle::memory::Copy(paddle::platform::CPUPlace(), host_data.data(),
                         paddle::platform::CUDAPlace(), ptr,
                         sizeof(T) * dense_t->numel(), stream);
    ptr = host_data.data();
  }

  for (int i = 0; i < dense_t->numel(); i++) {
    if (ptr[i] != value) return false;
  }
  return true;
}

inline void InitEnv(paddle::platform::Place place) {
  // Prepare Device Contexts
  // Init DeviceContextPool
  paddle::framework::InitDevices();

  // Init Tracer Place
  SetExpectedPlace(place);
}
}  // namespace egr
