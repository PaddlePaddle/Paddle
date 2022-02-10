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

#include "paddle/fluid/operators/fill_zeros_like_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FillZerosLikeNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");

    out->mutable_data<T>(context.GetPlace());
    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const auto& runner = NpuOpRunner("ZerosLike", {*x}, {*out});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    fill_zeros_like,
    ops::FillZerosLikeNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::FillZerosLikeNPUKernel<paddle::platform::NPUDeviceContext, int64_t>,
    ops::FillZerosLikeNPUKernel<paddle::platform::NPUDeviceContext,
                                paddle::platform::float16>,
    ops::FillZerosLikeNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::FillZerosLikeNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::FillZerosLikeNPUKernel<paddle::platform::NPUDeviceContext, bool>);
