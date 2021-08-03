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

#include "paddle/fluid/operators/mean_op.h"
#ifdef PADDLE_WITH_XPU
#include <memory>
#include <string>
#include <unordered_map>

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MeanGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto OG = context.Input<Tensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE_EQ(OG->numel(), 1, platform::errors::InvalidArgument(
                                          "Mean Gradient should be scalar"));
    auto IG = context.Output<Tensor>(framework::GradVarName("X"));
    IG->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    float* dx = IG->data<float>();
    const float* dy = OG->data<float>();
    int r = xpu::mean_grad(dev_ctx.x_context(), dx, dy, IG->numel());
    PADDLE_ENFORCE_EQ(
        r, xpu::Error_t::SUCCESS,
        platform::errors::External(
            "XPU kernel error. Mean_grad execution not succeed, error code=%d",
            r));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    mean_grad,
    ops::MeanGradXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
