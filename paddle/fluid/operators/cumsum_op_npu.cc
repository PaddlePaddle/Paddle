/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "paddle/fluid/operators/cum_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

void CheckAttrs(const framework::ExecutionContext& ctx) {
  // Add this check is is due to Ascend CumsumD does't supoort
  // attr flatten
  bool flatten = ctx.Attr<bool>("flatten");
  PADDLE_ENFORCE_EQ(flatten, false,
                    platform::errors::InvalidArgument(
                        "attr flatten must be false, but got true"));
}

template <typename DeviceContext, typename T>
class CumSumNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    LOG(WARNING) << "CumSumNPUKernel";

    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    int axis = ctx.Attr<int>("axis");
    // bool flatten = ctx.Attr<bool>("flatten");
    bool exclusive = ctx.Attr<bool>("exclusive");
    bool reverse = ctx.Attr<bool>("reverse");

    out->mutable_data<T>(ctx.GetPlace());

    framework::NPUAttributeMap attr_input = {
        {"axis", axis}, {"exclusive", exclusive}, {"reverse", reverse}};

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("CumsumD", {*x}, {*out}, attr_input);
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    cumsum, ops::CumSumNPUKernel<plat::NPUDeviceContext, float>,
    ops::CumSumNPUKernel<plat::NPUDeviceContext, plat::float16>);
