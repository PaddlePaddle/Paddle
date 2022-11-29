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

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/optimizers/pow2_decay_with_linear_warmup_op.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"

namespace paddle {
namespace operators {

template <typename T>
class Pow2DecayWithLinearWarmupXPUOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const {
    const auto *lr = ctx.Input<phi::DenseTensor>("LearningRate");
    const auto *step = ctx.Input<phi::DenseTensor>("Step");
    auto *lr_out = ctx.Output<phi::DenseTensor>("LearningRateOut");
    auto *step_out = ctx.Output<phi::DenseTensor>("StepOut");
    PADDLE_ENFORCE_EQ(
        lr,
        lr_out,
        platform::errors::InvalidArgument("Input(LearningRate) and "
                                          "Output(LearningRateOut) "
                                          "must be the same."));
    PADDLE_ENFORCE_NOT_NULL(lr,
                            platform::errors::InvalidArgument(
                                "Input(LearingRate) should not be nullptr."));
    PADDLE_ENFORCE_EQ(step,
                      step_out,
                      platform::errors::InvalidArgument(
                          "Input(Step) and Output(StepOut) must be the same."));
    PADDLE_ENFORCE_NOT_NULL(step,
                            platform::errors::InvalidArgument(
                                "Input(Step) should not be nullptr."));
    PADDLE_ENFORCE_EQ(
        step->IsInitialized(),
        true,
        platform::errors::InvalidArgument("Input(Step) must be initialized."));

    auto warmup_steps = static_cast<size_t>(ctx.Attr<int64_t>("warmup_steps"));
    auto total_steps = static_cast<size_t>(ctx.Attr<int64_t>("total_steps"));
    PADDLE_ENFORCE_LE(warmup_steps,
                      total_steps,
                      platform::errors::InvalidArgument(
                          "warmup_steps must not be larger than total_steps."));
    auto base_lr = ctx.Attr<float>("base_lr");
    auto end_lr = ctx.Attr<float>("end_lr");

    auto *lr_data = lr_out->data<T>();
    auto *step_data = step_out->data<int64_t>();
    auto &dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
    int r = xpu::pow2_decay_with_linear_warmup(dev_ctx.x_context(),
                                               lr_data,
                                               step_data,
                                               warmup_steps,
                                               total_steps,
                                               base_lr,
                                               end_lr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "pow2_decay_with_linear_warmup");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(pow2_decay_with_linear_warmup,
                       ops::Pow2DecayWithLinearWarmupXPUOpKernel<float>);
#endif
