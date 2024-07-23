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

#include "glog/logging.h"
#include "paddle/common/errors.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/kernels/scale_kernel.h"

namespace egr {

template <typename DeviceContext>
static void ScaleDeviceDispatch(const phi::DenseTensor& dense_tensor,
                                const DeviceContext& dev_ctx,
                                float scale,
                                float bias,
                                bool bias_after_scale,
                                phi::DenseTensor* dense_out) {
  switch (dense_tensor.dtype()) {
    case phi::DataType::FLOAT64: {
      phi::ScaleKernel<
          double,
          typename paddle::framework::ConvertToPhiContext<DeviceContext>::TYPE>(
          static_cast<const typename paddle::framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          dense_tensor /* tensor */,
          scale /* scale */,
          bias /* bias */,
          bias_after_scale /* bias_after_scale */,
          dense_out /* out tensor */);
      break;
    }
    case phi::DataType::FLOAT32: {
      phi::ScaleKernel<
          float,
          typename paddle::framework::ConvertToPhiContext<DeviceContext>::TYPE>(
          static_cast<const typename paddle::framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          dense_tensor /* tensor */,
          scale /* scale */,
          bias /* bias */,
          bias_after_scale /* bias_after_scale */,
          dense_out /* out tensor */);
      break;
    }
    case phi::DataType::INT64: {
      phi::ScaleKernel<
          int64_t,
          typename paddle::framework::ConvertToPhiContext<DeviceContext>::TYPE>(
          static_cast<const typename paddle::framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          dense_tensor /* tensor */,
          scale /* scale */,
          bias /* bias */,
          bias_after_scale /* bias_after_scale */,
          dense_out /* out tensor */);
      break;
    }
    case phi::DataType::INT32: {
      phi::ScaleKernel<
          int32_t,
          typename paddle::framework::ConvertToPhiContext<DeviceContext>::TYPE>(
          static_cast<const typename paddle::framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          dense_tensor /* tensor */,
          scale /* scale */,
          bias /* bias */,
          bias_after_scale /* bias_after_scale */,
          dense_out /* out tensor */);
      break;
    }
    default: {
      PADDLE_THROW(phi::errors::Fatal(
          "Detected unsupported data type."
          "Only Float64, Float32, Int64, Int32 are supported for now."));
      break;
    }
  }
}

void ScaleAPI(const paddle::Tensor& x,
              float scale,
              float bias,
              bool bias_after_scale,
              paddle::Tensor* out) {
  // TODO(jiabin): Support multiple tensor here, Create DenseTensor is not a
  // proper way to Demo it
  // Run Forward Function
  auto dense_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(x.impl());
  // Init output tensor
  auto tensor_meta = phi::DenseTensorMeta(
      dense_tensor->dtype(), dense_tensor->dims(), dense_tensor->layout());
  auto place = dense_tensor->place();
  size_t bytes_size =
      common::product(dense_tensor->dims()) * SizeOf(dense_tensor->dtype());
  auto dense_out = std::make_shared<phi::DenseTensor>(
      paddle::memory::Alloc(place, bytes_size), std::move(tensor_meta));
  // Handle Device Context
  const phi::Place& expected_kernel_place =
      Controller::Instance().GetExpectedPlace();
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();

  if (expected_kernel_place == phi::CPUPlace()) {
    auto* dev_ctx =
        dynamic_cast<phi::CPUContext*>(pool.Get(expected_kernel_place));
    if (!dev_ctx) {
      PADDLE_THROW(
          phi::errors::Fatal("Cannot convert device_context to phi::CPUContext."
                             "This indicates backend mismatch."
                             "Pleas double check your expected place"));
    }
    ScaleDeviceDispatch<phi::CPUContext>(*dense_tensor.get(),
                                         *dev_ctx,
                                         scale,
                                         bias,
                                         bias_after_scale,
                                         dense_out.get());

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (expected_kernel_place == phi::GPUPlace()) {
    auto* dev_ctx =
        dynamic_cast<phi::GPUContext*>(pool.Get(expected_kernel_place));
    if (!dev_ctx) {
      PADDLE_THROW(phi::errors::Fatal(
          "Cannot convert device_context to CUDADeviceContext."
          "This indicates backend mismatch."
          "Pleas double check your expected place"));
    }
    ScaleDeviceDispatch<phi::GPUContext>(*dense_tensor.get(),
                                         *dev_ctx,
                                         scale,
                                         bias,
                                         bias_after_scale,
                                         dense_out.get());
#endif
  } else {
    PADDLE_THROW(phi::errors::Fatal(
        "Detected unsupported backend."
        "Only CPU and CUDA Backend are supported for now."
        "Please double check if your backend falls into the above two "
        "categories."));
  }

  out->set_impl(dense_out);
}

void GradNodeScale::SetTensorWrappers_X(
    const std::vector<paddle::Tensor>& tensors) {
  // Does nothing for scale
}

void GradNodeScale::SetAttributes_scale(float scale) { scale_ = scale; }

paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>
GradNodeScale::operator()(
    paddle::small_vector<std::vector<paddle::Tensor>,
                         kSlotSmallVectorSize>& grads,  // NOLINT
    bool create_graph,
    bool is_new_grad) {
  // 1. Check Output Size
  VLOG(6) << "grad size is: " << grads.size();
  PADDLE_ENFORCE(
      ((grads.size() == 1) && (grads[0].size() == 1)),
      phi::errors::Fatal(
          "ScaleGradNode takes exactly 1 grad tensor."
          "However received: %d",
          "This indicates an issue with Eager Dygraph Backward logic",
          grads.size()));
  paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize> outs;
  // 2. Create needed out parttern
  paddle::Tensor out;
  // Apply Gradient Hooks
  if (GradientHooksRegistered()) {
    // TODO(jiabin): Shall we apply hook slot by slot here or accept
    // vector<vector<phi::tensor>> to apply all hooks?
    paddle::small_vector<std::vector<paddle::Tensor>, kSlotSmallVectorSize>
        hooked_grads = ApplyGradientHooks(grads);
    ScaleAPI(/* slot by slot set */ hooked_grads[0][0],
             scale_,
             0.0 /* bias */,
             true /* bias_after_scale */,
             &out);
  } else {
    ScaleAPI(
        grads[0][0], scale_, 0.0 /* bias */, true /* bias_after_scale */, &out);
  }

  return {{out}};
}

}  // namespace egr
