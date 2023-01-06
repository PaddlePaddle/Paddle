/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cmath>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ClearFloatStatusKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* float_status = ctx.Input<phi::DenseTensor>("FloatStatus");
    auto* float_status_out = ctx.Output<phi::DenseTensor>("FloatStatusOut");
    // NOTE(zhiqiu): NPUClearFloatStatus modifies the input.
    PADDLE_ENFORCE_EQ(float_status_out,
                      float_status,
                      platform::errors::PreconditionNotMet(
                          "The input(FloatStatus) and Output(FloatStatusOut) "
                          "should be the same."));
    phi::DenseTensor tmp;
    tmp.mutable_data<float>({8}, ctx.GetPlace());
    const auto& runner =
        NpuOpRunner("NPUClearFloatStatus", {tmp}, {*float_status_out});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    clear_float_status,
    ops::ClearFloatStatusKernel<paddle::platform::NPUDeviceContext, float>);
