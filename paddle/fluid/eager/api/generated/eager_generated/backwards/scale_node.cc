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

#include "paddle/phi/kernels/scale_kernel.h"

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

#include "glog/logging.h"

namespace egr {

template <typename DeviceContext>
static void ScaleDeviceDispatch(const phi::DenseTensor& dense_tensor,
                                const DeviceContext& dev_ctx, float scale,
                                float bias, bool bias_after_scale,
                                phi::DenseTensor* dense_out) {
  switch (dense_tensor.dtype()) {
    case phi::DataType::FLOAT64: {
      phi::ScaleKernel<double, typename paddle::framework::ConvertToPhiContext<
                                   DeviceContext>::TYPE>(
          static_cast<const typename paddle::framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          dense_tensor /* tensor */, scale /* scale */, bias /* bias */,
          bias_after_scale /* bias_after_scale */, dense_out /* out tensor */);
      break;
    }
    case phi::DataType::FLOAT32: {
      phi::ScaleKernel<float, typename paddle::framework::ConvertToPhiContext<
                                  DeviceContext>::TYPE>(
          static_cast<const typename paddle::framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          dense_tensor /* tensor */, scale /* scale */, bias /* bias */,
          bias_after_scale /* bias_after_scale */, dense_out /* out tensor */);
      break;
    }
    case phi::DataType::INT64: {
      phi::ScaleKernel<int64_t, typename paddle::framework::ConvertToPhiContext<
                                    DeviceContext>::TYPE>(
          static_cast<const typename paddle::framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          dense_tensor /* tensor */, scale /* scale */, bias /* bias */,
          bias_after_scale /* bias_after_scale */, dense_out /* out tensor */);
      break;
    }
    case phi::DataType::INT32: {
      phi::ScaleKernel<int32_t, typename paddle::framework::ConvertToPhiContext<
                                    DeviceContext>::TYPE>(
          static_cast<const typename paddle::framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          dense_tensor /* tensor */, scale /* scale */, bias /* bias */,
          bias_after_scale /* bias_after_scale */, dense_out /* out tensor */);
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

void ScaleAPI(const paddle::experimental::Tensor& x, float scale, float bias,
              bool bias_after_scale, paddle::experimental::Tensor* out) {
  // TODO(jiabin): Support multiple tensor here, Create DenseTensor is not a
  // proper way to Demo it
  // Run Forward Function
  auto dense_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(x.impl());
  // Init output tensor
  auto tensor_meta = phi::DenseTensorMeta(
      dense_tensor->dtype(), dense_tensor->dims(), dense_tensor->layout());
  auto place = dense_tensor->place();
  size_t bytes_size =
      phi::product(dense_tensor->dims()) * SizeOf(dense_tensor->dtype());
  auto dense_out = std::make_shared<phi::DenseTensor>(
      phi::make_intrusive<paddle::experimental::SharedStorage>(
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
    const std::vector<paddle::experimental::Tensor>& tensors) {
  // Does nothing for scale
}

void GradNodeScale::SetAttributes_scale(float scale) { scale_ = scale; }

std::vector<std::vector<paddle::experimental::Tensor>> GradNodeScale::
operator()(
    std::vector<std::vector<paddle::experimental::Tensor>>& grads,  // NOLINT
    bool create_graph) {
  // 1. Check Output Size
  PADDLE_ENFORCE(
      ((grads.size() == 1) && (grads[0].size() == 1)),
      paddle::platform::errors::Fatal(
          "ScaleGradNode takes exactly 1 grad tensor."
          "However received: %d",
          "This indicates an issue with Eager Dygraph Backward logic",
          grads.size()));
  std::vector<std::vector<paddle::experimental::Tensor>> outs;
  // 2. Create needed out parttern
  paddle::experimental::Tensor out;
  // Apply Gradient Hooks
  if (GradientHooksRegistered()) {
    // TODO(jiabin): Shall we apply hook slot by slot here or accept
    // vector<vector<phi::tensor>> to apply all hooks?
    std::vector<std::vector<paddle::experimental::Tensor>> hooked_grads =
        ApplyGradientHooks(grads);
    ScaleAPI(/* slot by slot set */ hooked_grads[0][0], scale_, 0.0 /* bias */,
             true /* bias_after_scale */, &out);
  } else {
    ScaleAPI(grads[0][0], scale_, 0.0 /* bias */, true /* bias_after_scale */,
             &out);
  }

  return {{out}};
}

}  // namespace egr
