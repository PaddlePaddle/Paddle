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

#include <memory>
#include <string>

#include "paddle/fluid/operators/mul_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SizeNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    VLOG(3) << "Ensure the op execute on NPU" << std::endl;
    auto* x = ctx.Input<framework::Tensor>("Input");
    auto* out = ctx.Output<framework::Tensor>("Out");

    framework::NPUAttributeMap attr_input = {};
    // set attrs if have
    if (ctx.HasAttr("out_type")) {
      attr_input["out_type"] = ctx.Attr<int>("out_type");
    }

    out->mutable_data<T>(ctx.GetPlace());
    const auto& runner = NpuOpRunner("Size", {*x}, {*out}, attr_input);
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
    size, ops::SizeNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::SizeNPUKernel<paddle::platform::NPUDeviceContext, int64_t>,
    ops::SizeNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>,
    ops::SizeNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::SizeNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::SizeNPUKernel<paddle::platform::NPUDeviceContext, bool>);

