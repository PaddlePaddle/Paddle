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

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/utils.h"

#include "paddle/pten/api/all.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/tensor_meta.h"

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"

namespace eager_test {

template <typename T>
bool CompareGradTensorWithValue(const paddle::experimental::Tensor& target,
                                T value) {
  egr::AutogradMeta* meta = egr::EagerUtils::unsafe_autograd_meta(target);
  auto grad_dense =
      std::dynamic_pointer_cast<pten::DenseTensor>(meta->Grad().impl());
  T* ptr = grad_dense->data<T>();

  std::vector<T> host_data(grad_dense->numel());
  if (paddle::platform::is_gpu_place(grad_dense->place())) {
#ifdef PADDLE_WITH_CUDA
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx = dynamic_cast<paddle::platform::CUDADeviceContext*>(
        pool.Get(paddle::platform::CUDAPlace()));
    auto stream = dev_ctx->stream();

    paddle::memory::Copy(paddle::platform::CPUPlace(), host_data.data(),
                         paddle::platform::CUDAPlace(), ptr,
                         sizeof(T) * grad_dense->numel(), stream);
    ptr = host_data.data();
#endif
  }
  VLOG(6) << "CompareGradTensorWithValue";
  for (int i = 0; i < grad_dense->numel(); i++) {
    PADDLE_ENFORCE(value == ptr[i],
                   paddle::platform::errors::PreconditionNotMet(
                       "Numerical Error in Compare Grad Variable With Value of "
                       "%d, we expected got value: %f, but got: %f instead. "
                       "Please check it later.",
                       i, value, ptr[i]));
  }
  return true;
}

template <typename T>
bool CompareTensorWithValue(const paddle::experimental::Tensor& target,
                            T value) {
  // TODO(jiabin): Support Selected Rows later
  auto dense_t = std::dynamic_pointer_cast<pten::DenseTensor>(target.impl());
  T* ptr = dense_t->data<T>();

  std::vector<T> host_data(dense_t->numel());
  if (paddle::platform::is_gpu_place(dense_t->place())) {
#ifdef PADDLE_WITH_CUDA
    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    auto* dev_ctx = dynamic_cast<paddle::platform::CUDADeviceContext*>(
        pool.Get(paddle::platform::CUDAPlace()));
    auto stream = dev_ctx->stream();

    paddle::memory::Copy(paddle::platform::CPUPlace(), host_data.data(),
                         paddle::platform::CUDAPlace(), ptr,
                         sizeof(T) * dense_t->numel(), stream);
    ptr = host_data.data();
#endif
  }

  VLOG(6) << "CompareTensorWithValue";
  for (int i = 0; i < dense_t->numel(); i++) {
    PADDLE_ENFORCE(value == ptr[i],
                   paddle::platform::errors::PreconditionNotMet(
                       "Numerical Error in Compare Grad Variable With Value of "
                       "%d, we expected got value: %f, but got: %f instead. "
                       "Please check it later.",
                       i, value, ptr[i]));
  }
  return true;
}

inline void InitEnv(paddle::platform::Place place) {
  // Prepare Device Contexts
  // Init DeviceContextPool
  paddle::framework::InitDevices();

  // Init Tracer Place
  egr::Controller::Instance().SetExpectedPlace(place);
}

}  // namespace eager_test
