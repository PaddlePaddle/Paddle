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

#include "paddle/fluid/operators/arg_min_max_op_base.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {
using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class ArgMinNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    int64_t axis = ctx.Attr<int64_t>("axis");
    auto dtype = ctx.Attr<int>("dtype");

    auto* out = ctx.Output<phi::DenseTensor>("Out");
    out->mutable_data<int32_t>(ctx.GetPlace());

    NpuOpRunner runner;
    runner.SetType("ArgMin")
        .AddInput(*x)
        .AddInput(std::vector<int64_t>{axis})
        .AddOutput(*out)
        .AddAttr("dtype", dtype);

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
    arg_min,
    ops::ArgMinNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ArgMinNPUKernel<paddle::platform::NPUDeviceContext,
                         paddle::platform::float16>);
