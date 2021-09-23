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

#include "paddle/tcmpt/api/all.h"
#include "paddle/tcmpt/core/dense_tensor.h"

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"

namespace egr {

static std::shared_ptr<paddle::platform::Place> _expected_place(nullptr);

const paddle::platform::Place& GetExpectedPlace() {
  return *_expected_place.get();
}

void SetExpectedPlace(const paddle::platform::Place& place) {
  _expected_place = std::make_shared<paddle::platform::Place>(place);
}

template <typename DeviceContext>
static void ScaleDeviceDispatch(const pt::DenseTensor& dense_tensor,
                                const DeviceContext& dev_ctx, float scale,
                                float bias, bool bias_after_scale,
                                pt::DenseTensor* dense_out) {
  switch (dense_tensor.type()) {
    case pt::DataType::kFLOAT64: {
      pt::Scale<double>(dev_ctx, dense_tensor /* tensor */, scale /* scale */,
                        bias /* bias */,
                        bias_after_scale /* bias_after_scale */,
                        dense_out /* out tensor */);
      break;
    }
    case pt::DataType::kFLOAT32: {
      pt::Scale<float>(dev_ctx, dense_tensor /* tensor */, scale /* scale */,
                       bias /* bias */, bias_after_scale /* bias_after_scale */,
                       dense_out /* out tensor */);
      break;
    }
    case pt::DataType::kINT64: {
      pt::Scale<int64_t>(dev_ctx, dense_tensor /* tensor */, scale /* scale */,
                         bias /* bias */,
                         bias_after_scale /* bias_after_scale */,
                         dense_out /* out tensor */);
      break;
    }
    case pt::DataType::kINT32: {
      pt::Scale<int32_t>(dev_ctx, dense_tensor /* tensor */, scale /* scale */,
                         bias /* bias */,
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
static void FillConstCPUFunctor(pt::DenseTensor* tensor_dense, double value) {
  PADDLE_ENFORCE(tensor_dense, paddle::platform::errors::Fatal(
                                   "Receive nullptr of dense tensor"));
  switch (tensor_dense->type()) {
    case pt::DataType::kINT64: {
      int64_t* data_ptr = tensor_dense->mutable_data<int64_t>();
      for (int i = 0; i < tensor_dense->numel(); i++) {
        data_ptr[i] = static_cast<int64_t>(value);
      }

      break;
    }
    case pt::DataType::kINT32: {
      int32_t* data_ptr = tensor_dense->mutable_data<int32_t>();
      for (int i = 0; i < tensor_dense->numel(); i++) {
        data_ptr[i] = static_cast<int32_t>(value);
      }

      break;
    }
    case pt::DataType::kFLOAT64: {
      double* data_ptr = tensor_dense->mutable_data<double>();
      for (int i = 0; i < tensor_dense->numel(); i++) {
        data_ptr[i] = static_cast<double>(value);
      }

      break;
    }
    case pt::DataType::kFLOAT32: {
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
static void FillConstCUDAFunctor(pt::DenseTensor* tensor_dense, double value) {
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = dynamic_cast<paddle::platform::CUDADeviceContext*>(
      pool.Get(paddle::platform::CUDAPlace()));
  auto stream = dev_ctx->stream();

  switch (tensor_dense->type()) {
    case pt::DataType::kINT64: {
      std::vector<int64_t> host_data(tensor_dense->numel(),
                                     static_cast<int64_t>(value));
      int64_t* device_ptr = tensor_dense->mutable_data<int64_t>();
      paddle::memory::Copy(paddle::platform::CUDAPlace(), device_ptr,
                           paddle::platform::CPUPlace(), host_data.data(),
                           sizeof(int64_t) * tensor_dense->numel(), stream);
      break;
    }
    case pt::DataType::kINT32: {
      std::vector<int32_t> host_data(tensor_dense->numel(),
                                     static_cast<int32_t>(value));
      int32_t* device_ptr = tensor_dense->mutable_data<int32_t>();
      paddle::memory::Copy(paddle::platform::CUDAPlace(), device_ptr,
                           paddle::platform::CPUPlace(), host_data.data(),
                           sizeof(int32_t) * tensor_dense->numel(), stream);
      break;
    }
    case pt::DataType::kFLOAT64: {
      std::vector<double> host_data(tensor_dense->numel(),
                                    static_cast<double>(value));
      double* device_ptr = tensor_dense->mutable_data<double>();
      paddle::memory::Copy(paddle::platform::CUDAPlace(), device_ptr,
                           paddle::platform::CPUPlace(), host_data.data(),
                           sizeof(double) * tensor_dense->numel(), stream);
      break;
    }
    case pt::DataType::kFLOAT32: {
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

void ScaleAPI(const pt::Tensor& x, float scale, float bias,
              bool bias_after_scale, pt::Tensor* out) {
  // TODO(jiabin): Support multiple tensor here, Create DenseTensor is not a
  // proper way to Demo it
  // Run Forward Function
  auto dense_tensor = std::dynamic_pointer_cast<pt::DenseTensor>(x.impl());

  // Init output tensor
  auto tensor_meta =
      pt::TensorMeta(dense_tensor->dims(), dense_tensor->backend(),
                     dense_tensor->type(), dense_tensor->layout());
  auto dense_out = std::make_shared<pt::DenseTensor>(std::move(tensor_meta),
                                                     pt::TensorStatus());

  // Handle Device Context
  const paddle::platform::Place& expected_kernel_place = GetExpectedPlace();
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

  out->SetImpl(dense_out);
}

void FillConstAPI(double value, const pt::DDim& ddim,
                  const pt::Backend& backend, const pt::DataType& dtype,
                  const pt::DataLayout& layout, pt::Tensor* target) {
  // Create new tensor->impl and fill it with 1.0
  // Fill 1.0
  std::shared_ptr<pt::DenseTensor> tensor_dense = nullptr;
  if (!target->defined() || !target->initialized()) {
    VLOG(6) << "Init undefined or uninitialized tensor in FillConstAPI";
    auto tensor_meta = pt::TensorMeta(ddim, backend, dtype, layout);
    tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta),
                                                     pt::TensorStatus());
    target->SetImpl(tensor_dense);

  } else {
    tensor_dense = std::dynamic_pointer_cast<pt::DenseTensor>(target->impl());
  }

  if (!tensor_dense) {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "FillConstAPI Only supports InputBuffer with DenseTensor for now."));
  }
  VLOG(6) << "Call FillConstKernel";
  switch (tensor_dense->backend()) {
    case pt::Backend::kCPU: {
      VLOG(8) << "Call FillConst CPU Kernel";
      FillConstCPUFunctor(tensor_dense.get(), value);
      break;
    }
    case pt::Backend::kCUDA: {
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

}  // namespace egr
