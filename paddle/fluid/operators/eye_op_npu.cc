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

#include "paddle/fluid/operators/eye_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class EyeNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto num_rows = ctx.Attr<int64_t>("num_rows");

    auto d_nums = ctx.Attr<int>("dtype");
    auto dtype =
        ConvertToNpuDtype(static_cast<framework::proto::VarType::Type>(d_nums));

    auto num_columns = ctx.Attr<int64_t>("num_columns");
    if (num_columns == -1) num_columns = num_rows;

    framework::NPUAttributeMap attr_input = {
        {"num_rows", num_rows}, {"num_columns", num_columns}, {"dtype", dtype}};

    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    const auto& runner = NpuOpRunner("Eye", {}, {*out}, attr_input);
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
    eye, ops::EyeNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::EyeNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::EyeNPUKernel<paddle::platform::NPUDeviceContext,
                      paddle::platform::float16>);
