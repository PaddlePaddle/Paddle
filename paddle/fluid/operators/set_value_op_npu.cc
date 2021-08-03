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

#include "paddle/fluid/operators/mean_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SetValueNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* inp = ctx.Input<framework::LoDTensor>("Input");
    auto* out = ctx.Output<framework::LoDTensor>("Out");

    std::vector<int> axes;

    framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                             {"axes", axes}};

    out->mutable_data<T>(ctx.GetPlace());

    const auto& runner = NpuOpRunner("ReduceMeanD", {*x}, {*out}, attr_input);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    mean, ops::MeanNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::MeanNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MeanNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::MeanNPUKernel<paddle::platform::NPUDeviceContext, plat::float16>)

REGISTER_OP_NPU_KERNEL(
    mean_grad, ops::MeanGradNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::MeanGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MeanGradNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::MeanGradNPUKernel<paddle::platform::NPUDeviceContext, plat::float16>)

