/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
template <typename DeviceContext, typename T>
class AccuracyXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* inference = ctx.Input<phi::DenseTensor>("Out");
    auto* indices = ctx.Input<phi::DenseTensor>("Indices");
    auto* label = ctx.Input<phi::DenseTensor>("Label");
    auto* accuracy = ctx.Output<phi::DenseTensor>("Accuracy");
    auto* correct = ctx.Output<phi::DenseTensor>("Correct");
    auto* total = ctx.Output<phi::DenseTensor>("Total");
    int* correct_data = correct->mutable_data<int>(ctx.GetPlace());
    int* total_data = total->mutable_data<int>(ctx.GetPlace());
    float* accuracy_data = accuracy->mutable_data<float>(ctx.GetPlace());
    const int64_t* indices_data = indices->data<int64_t>();
    const int64_t* label_data = label->data<int64_t>();
    size_t num_samples = inference->dims()[0];
    size_t class_dim = inference->dims()[1];
    if (num_samples == 0) {
      return;
    }
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    int size = num_samples * class_dim;
    int* indices_int32_ptr = RAII_GUARD.alloc_l3_or_gm<int>(size);
    PADDLE_ENFORCE_XDNN_NOT_NULL(indices_int32_ptr);
    int* label_int32_ptr = RAII_GUARD.alloc_l3_or_gm<int>(size);
    PADDLE_ENFORCE_XDNN_NOT_NULL(label_int32_ptr);

    int r = xpu::cast_v2<int64_t, int32_t>(
        dev_ctx.x_context(), indices_data, indices_int32_ptr, size);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast_v2");

    r = xpu::cast_v2<int64_t, int32_t>(
        dev_ctx.x_context(), label_data, label_int32_ptr, size);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast_v2");

    r = xpu::accuracy(dev_ctx.x_context(),
                      indices_int32_ptr,
                      label_int32_ptr,
                      num_samples,
                      class_dim,
                      correct_data,
                      total_data,
                      accuracy_data);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast_v2");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    accuracy,
    ops::AccuracyXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
