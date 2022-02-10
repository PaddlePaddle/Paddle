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

#include "paddle/fluid/operators/mul_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SizeNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("Input");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    Tensor cpu_tensor;
    auto cpu_data =
        cpu_tensor.mutable_data<int64_t>(out->dims(), platform::CPUPlace());
    cpu_data[0] = x->numel();
    paddle::framework::TensorCopy(
        cpu_tensor, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), out);
    ctx.template device_context<paddle::platform::NPUDeviceContext>().Wait();
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
