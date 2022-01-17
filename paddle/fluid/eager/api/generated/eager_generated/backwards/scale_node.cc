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

#include "paddle/fluid/eager/api/generated/eager_generated/backwards/scale_node.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/eager_tensor.h"

#include "paddle/pten/kernels/scale_kernel.h"

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

#include "glog/logging.h"

namespace egr {

template <typename DeviceContext>
static void ScaleDeviceDispatch(const pten::DenseTensor& dense_tensor,
                                const DeviceContext& dev_ctx, float scale,
                                float bias, bool bias_after_scale,
                                pten::DenseTensor* dense_out) {
  switch (dense_tensor.dtype()) {
    case pten::DataType::FLOAT64: {
      pten::ScaleKernel<double, DeviceContext>(
          dev_ctx, dense_tensor /* tensor */, scale /* scale */,
          bias /* bias */, bias_after_scale /* bias_after_scale */,
          dense_out /* out tensor */);
      break;
    }
    case pten::DataType::FLOAT32: {
      pten::ScaleKernel<float, DeviceContext>(
          dev_ctx, dense_tensor /* tensor */, scale /* scale */,
          bias /* bias */, bias_after_scale /* bias_after_scale */,
          dense_out /* out tensor */);
      break;
    }
    case pten::DataType::INT64: {
      pten::ScaleKernel<int64_t, DeviceContext>(
          dev_ctx, dense_tensor /* tensor */, scale /* scale */,
          bias /* bias */, bias_after_scale /* bias_after_scale */,
          dense_out /* out tensor */);
      break;
    }
    case pten::DataType::INT32: {
      pten::ScaleKernel<int32_t, DeviceContext>(
          dev_ctx, dense_tensor /* tensor */, scale /* scale */,
          bias /* bias */, bias_after_scale /* bias_after_scale */,
          dense_out /* out tensor */);
      break;
    }
    default: {
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Detected unsupported data type."
          "Only Float64, Float32, Int64, Int32 are supported for now."));
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
  auto tensor_meta = pten::DenseTensorMeta(
      dense_tensor->dtype(), dense_tensor->dims(), dense_tensor->layout());
  auto place = dense_tensor->place();
  size_t bytes_size = paddle::framework::product(dense_tensor->dims()) *
                      SizeOf(dense_tensor->dtype());
  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          paddle::memory::Alloc(place, bytes_size)),
      std::move(tensor_meta));
  // Handle Device Context
  const paddle::platform::Place& expected_kernel_place =
      Controller::Instance().GetExpectedPlace();
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();

  if (expected_kernel_place == paddle::platform::CPUPlace()) {
    auto* dev_ctx = dynamic_cast<paddle::platform::CPUDeviceContext*>(
        pool.Get(expected_kernel_place));
    if (!dev_ctx) {
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Cannot convert device_context to CPUDeviceContext."
          "This indicates backend mismatch."
          "Pleas double check your expected place"));
    }
    ScaleDeviceDispatch<paddle::platform::CPUDeviceContext>(
        *dense_tensor.get(), *dev_ctx, scale, bias, bias_after_scale,
        dense_out.get());

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (expected_kernel_place == paddle::platform::CUDAPlace()) {
    auto* dev_ctx = dynamic_cast<paddle::platform::CUDADeviceContext*>(
        pool.Get(expected_kernel_place));
    if (!dev_ctx) {
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Cannot convert device_context to CUDADeviceContext."
          "This indicates backend mismatch."
          "Pleas double check your expected place"));
    }
    ScaleDeviceDispatch<paddle::platform::CUDADeviceContext>(
        *dense_tensor.get(), *dev_ctx, scale, bias, bias_after_scale,
        dense_out.get());
#endif
  } else {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Detected unsupported backend."
        "Only CPU and CUDA Backend are supported for now."
        "Please double check if your backend falls into the above two "
        "categories."));
  }

  out->set_impl(dense_out);
}

void GradNodeScale::SetTensorWrappers_X(
    const std::vector<egr::EagerTensor>& tensors) {
  // Does nothing for scale
}

void GradNodeScale::SetAttributes_scale(float scale) { scale_ = scale; }

std::vector<std::vector<egr::EagerTensor>> GradNodeScale::operator()(
    const std::vector<std::vector<egr::EagerTensor>>& grads) {
  // 1. Check Output Size
  PADDLE_ENFORCE(
      ((grads.size() == 1) && (grads[0].size() == 1)),
      paddle::platform::errors::Fatal(
          "ScaleGradNode takes exactly 1 grad tensor."
          "However received: %d",
          "This indicates an issue with Eager Dygraph Backward logic",
          grads.size()));
  std::vector<std::vector<egr::EagerTensor>> outs;
  // 2. Create needed out parttern
  egr::EagerTensor out;
  // Apply Gradient Hooks
  if (GradientHooksRegistered()) {
    // TODO(jiabin): Shall we apply hook slot by slot here or accept
    // vector<vector<pten::tensor>> to apply all hooks?
    std::vector<std::vector<egr::EagerTensor>> hooked_grads =
        ApplyGradientHooks(grads);
    ScaleAPI(/* slot by slot set */ hooked_grads[0][0], scale_, 0.0 /* bias */,
             true /* bias_after_scale */, &out);
  } else {
    ScaleAPI(grads[0][0], scale_, 0.0 /* bias */, true /* bias_after_scale */,
             &out);
  }

  // Apply Reduce Hooks
  if (ReduceHooksRegistered()) {
    ApplyReduceHooks();
  }
  return {{out}};
}

}  // namespace egr
