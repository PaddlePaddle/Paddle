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

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class GetFloatStatusKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* float_status = ctx.Input<framework::Tensor>("FloatStatus");
    auto* float_status_out = ctx.Output<framework::Tensor>("FloatStatusOut");
    // GetClearFloatStatus modifies the input.
    PADDLE_ENFORCE_EQ(float_status_out, float_status,
                      platform::errors::PreconditionNotMet(
                          "The input(FloatStatus) and Output(FloatStatusOut) "
                          "should be the same."));
    Tensor tmp;
    tmp.mutable_data<float>({8}, ctx.GetPlace());
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    // NPUGetFloatStatus updates data on input in-place.
    // tmp is only placeholder.
    NpuOpRunner("NPUGetFloatStatus", {*float_status}, {tmp}).Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    get_float_status,
    ops::GetFloatStatusKernel<paddle::platform::NPUDeviceContext, float>);
