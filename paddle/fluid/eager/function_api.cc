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

#include "paddle/fluid/eager/function_api.h"

#include "paddle/pten/api/all.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/hapi/all.h"

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device_context.h"

namespace egr {

Controller* Controller::controller_ = new Controller();

template <typename DeviceContext>
static void ScaleDeviceDispatch(const pten::DenseTensor& dense_tensor,
                                const DeviceContext& dev_ctx, float scale,
                                float bias, bool bias_after_scale,
                                pten::DenseTensor* dense_out) {
  switch (dense_tensor.data_type()) {
    case pten::DataType::FLOAT64: {
      pten::Scale<double>(dev_ctx, dense_tensor /* tensor */, scale /* scale */,
                          bias /* bias */,
                          bias_after_scale /* bias_after_scale */,
                          dense_out /* out tensor */);
      break;
    }
    case pten::DataType::FLOAT32: {
      pten::Scale<float>(dev_ctx, dense_tensor /* tensor */, scale /* scale */,
                         bias /* bias */,
                         bias_after_scale /* bias_after_scale */,
                         dense_out /* out tensor */);
      break;
    }
    case pten::DataType::INT64: {
      pten::Scale<int64_t>(dev_ctx, dense_tensor /* tensor */,
                           scale /* scale */, bias /* bias */,
                           bias_after_scale /* bias_after_scale */,
                           dense_out /* out tensor */);
      break;
    }
    case pten::DataType::INT32: {
      pten::Scale<int32_t>(dev_ctx, dense_tensor /* tensor */,
                           scale /* scale */, bias /* bias */,
                           bias_after_scale /* bias_after_scale */,
                           dense_out /* out tensor */);
      break;
    }
    default: {
      PADDLE_THROW(paddle::platform::errors::Fatal("Unsupported data type"));
      break;
    }
  }
}

// TODO(jiabin): This may have serious performance issue move code from
// gradient_accumulator.cc
static void FillConstCPUFunctor(pten::DenseTensor* tensor_dense, double value) {
  PADDLE_ENFORCE(tensor_dense, paddle::platform::errors::Fatal(
                                   "Receive nullptr of dense tensor"));
  switch (tensor_dense->data_type()) {
    case pten::DataType::INT64: {
      int64_t* data_ptr = tensor_dense->mutable_data<int64_t>();
      for (int i = 0; i < tensor_dense->numel(); i++) {
        data_ptr[i] = static_cast<int64_t>(value);
      }

      break;
    }
    case pten::DataType::INT32: {
      int32_t* data_ptr = tensor_dense->mutable_data<int32_t>();
      for (int i = 0; i < tensor_dense->numel(); i++) {
        data_ptr[i] = static_cast<int32_t>(value);
      }

      break;
    }
    case pten::DataType::FLOAT64: {
      double* data_ptr = tensor_dense->mutable_data<double>();
      for (int i = 0; i < tensor_dense->numel(); i++) {
        data_ptr[i] = static_cast<double>(value);
      }

      break;
    }
    case pten::DataType::FLOAT32: {
      float* data_ptr = tensor_dense->mutable_data<float>();
      for (int i = 0; i < tensor_dense->numel(); i++) {
        data_ptr[i] = static_cast<float>(value);
      }
      break;
    }
    default: {
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Only supports tensor with fp32, fp64, int32, int64 datatypes for "
          "now"));
      break;
    }
  }
}

// TODO(jiabin): This may have serious performance issue move code from
// gradient_accumulator.cc
static void FillConstCUDAFunctor(pten::DenseTensor* tensor_dense,
                                 double value) {
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = dynamic_cast<paddle::platform::CUDADeviceContext*>(
      pool.Get(paddle::platform::CUDAPlace()));
  auto stream = dev_ctx->stream();

  switch (tensor_dense->data_type()) {
    case pten::DataType::INT64: {
      std::vector<int64_t> host_data(tensor_dense->numel(),
                                     static_cast<int64_t>(value));
      int64_t* device_ptr = tensor_dense->mutable_data<int64_t>();
      paddle::memory::Copy(paddle::platform::CUDAPlace(), device_ptr,
                           paddle::platform::CPUPlace(), host_data.data(),
                           sizeof(int64_t) * tensor_dense->numel(), stream);
      break;
    }
    case pten::DataType::INT32: {
      std::vector<int32_t> host_data(tensor_dense->numel(),
                                     static_cast<int32_t>(value));
      int32_t* device_ptr = tensor_dense->mutable_data<int32_t>();
      paddle::memory::Copy(paddle::platform::CUDAPlace(), device_ptr,
                           paddle::platform::CPUPlace(), host_data.data(),
                           sizeof(int32_t) * tensor_dense->numel(), stream);
      break;
    }
    case pten::DataType::FLOAT64: {
      std::vector<double> host_data(tensor_dense->numel(),
                                    static_cast<double>(value));
      double* device_ptr = tensor_dense->mutable_data<double>();
      paddle::memory::Copy(paddle::platform::CUDAPlace(), device_ptr,
                           paddle::platform::CPUPlace(), host_data.data(),
                           sizeof(double) * tensor_dense->numel(), stream);
      break;
    }
    case pten::DataType::FLOAT32: {
      std::vector<float> host_data(tensor_dense->numel(),
                                   static_cast<float>(value));
      float* device_ptr = tensor_dense->mutable_data<float>();
      paddle::memory::Copy(paddle::platform::CUDAPlace(), device_ptr,
                           paddle::platform::CPUPlace(), host_data.data(),
                           sizeof(float) * tensor_dense->numel(), stream);
      break;
    }
    default: {
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Only supports tensor with fp32, fp64, int32, int64 datatypes for "
          "now"));
      break;
    }
  }
}

void ScaleAPI(const egr::EagerTensor& x, float scale, float bias,
              bool bias_after_scale, egr::EagerTensor* out) {
  // TODO(jiabin): Support multiple tensor here, Create DenseTensor is not a
  // proper way to Demo it
  // Run Forward Function
  auto dense_tensor = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());

  // Init output tensor
  auto tensor_meta =
      pten::TensorMeta(dense_tensor->dims(), dense_tensor->backend(),
                       dense_tensor->data_type(), dense_tensor->layout());
  auto dense_out = std::make_shared<pten::DenseTensor>(std::move(tensor_meta),
                                                       pten::TensorStatus());

  // Handle Device Context
  const paddle::platform::Place& expected_kernel_place =
      Controller::Instance().GetExpectedPlace();
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();

  if (expected_kernel_place == paddle::platform::CPUPlace()) {
    auto* dev_ctx = dynamic_cast<paddle::platform::CPUDeviceContext*>(
        pool.Get(expected_kernel_place));
    if (!dev_ctx) {
      PADDLE_THROW(paddle::platform::errors::Fatal("Backend mismatch"));
    }
    ScaleDeviceDispatch<paddle::platform::CPUDeviceContext>(
        *dense_tensor.get(), *dev_ctx, scale, bias, bias_after_scale,
        dense_out.get());

  } else if (expected_kernel_place == paddle::platform::CUDAPlace()) {
    auto* dev_ctx = dynamic_cast<paddle::platform::CUDADeviceContext*>(
        pool.Get(expected_kernel_place));
    if (!dev_ctx) {
      PADDLE_THROW(paddle::platform::errors::Fatal("Backend mismatch"));
    }
    ScaleDeviceDispatch<paddle::platform::CUDADeviceContext>(
        *dense_tensor.get(), *dev_ctx, scale, bias, bias_after_scale,
        dense_out.get());

  } else {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Only CPU and CUDA Backend are supported for now"));
  }

  out->set_impl(dense_out);
}

void FillConstAPI(double value, const pten::DDim& ddim,
                  const pten::Backend& backend, const pten::DataType& dtype,
                  const pten::DataLayout& layout, egr::EagerTensor* target) {
  // Create new tensor->impl and fill it with 1.0
  // Fill 1.0
  // TODO(jiabin): Refactor this with operators::math::set_constant
  std::shared_ptr<pten::DenseTensor> tensor_dense = nullptr;
  if (!target->defined() || !target->initialized()) {
    VLOG(6) << "Init undefined or uninitialized tensor in FillConstAPI";
    auto tensor_meta = pten::TensorMeta(ddim, backend, dtype, layout);
    tensor_dense = std::make_shared<pten::DenseTensor>(std::move(tensor_meta),
                                                       pten::TensorStatus());
    target->set_impl(tensor_dense);

  } else {
    tensor_dense = std::dynamic_pointer_cast<pten::DenseTensor>(target->impl());
  }

  if (!tensor_dense) {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "FillConstAPI Only supports InputBuffer with DenseTensor for now."));
  }
  VLOG(6) << "Call FillConstKernel";
  switch (tensor_dense->backend()) {
    case pten::Backend::CPU: {
      VLOG(8) << "Call FillConst CPU Kernel";
      FillConstCPUFunctor(tensor_dense.get(), value);
      break;
    }
    case pten::Backend::CUDA: {
      VLOG(8) << "Call FillConst CUDA Kernel";
      FillConstCUDAFunctor(tensor_dense.get(), value);
      break;
    }
    default: {
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Only CPU and CUDA Backend are supported for now"));
    }
  }
}

void FillConstAPI(double value, const paddle::framework::DDim& ddim,
                  const paddle::platform::Place& place,
                  const paddle::framework::proto::VarType::Type& dtype,
                  egr::EagerTensor* target) {
  auto* dst_tensor =
      target->MutableVar()->GetMutable<paddle::framework::LoDTensor>();
  auto* dev_ctx = paddle::platform::DeviceContextPool::Instance().Get(place);
  dst_tensor->Resize(ddim);
  // TOOD(jiabin): Ugly fix here we have fwd_data_type_ and data_type, since in
  // grad mission
  // we can't get data_type_ directly. We need to check if we can only use
  // default data_type for now.
  dst_tensor->mutable_data(place, dtype);
  paddle::operators::math::set_constant(*dev_ctx, dst_tensor, value);
}

}  // namespace egr
