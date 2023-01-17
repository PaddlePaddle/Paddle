/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class ShapeNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("Input");
    auto* out_t = ctx.Output<phi::DenseTensor>("Out");
    out_t->Resize({x->dims().size()});
    out_t->mutable_data<int32_t>(ctx.GetPlace());

    // The output data type defaults to int32.
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    NpuOpRunner runner;
    auto dst_dtype = ConvertToNpuDtype(framework::proto::VarType::INT32);
    runner.SetType("Shape").AddInput(*x).AddOutput(*out_t).AddAttr(
        "dtype", static_cast<int>(dst_dtype));
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    shape,
    ops::ShapeNPUKernel<paddle::platform::NPUDeviceContext, bool>,
    ops::ShapeNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::ShapeNPUKernel<paddle::platform::NPUDeviceContext, int8_t>,
    ops::ShapeNPUKernel<paddle::platform::NPUDeviceContext, uint8_t>,
    ops::ShapeNPUKernel<paddle::platform::NPUDeviceContext, int64_t>,
    ops::ShapeNPUKernel<paddle::platform::NPUDeviceContext,
                        paddle::platform::float16>,
    ops::ShapeNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ShapeNPUKernel<paddle::platform::NPUDeviceContext, double>);
