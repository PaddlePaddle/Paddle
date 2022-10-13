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

// TODO(Aganlengzi): delete this macro control and remove REMOVE_ITEM in
// cmake/operators.cmake when Paddle supports
#if (CANN_VERSION_CODE >= 504000)

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class NPUMultinomialKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    const int64_t num_samples = ctx.Attr<int>("num_samples");
    const bool replacement = ctx.Attr<bool>("replacement");

    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    out->mutable_data<int64_t>(place);

    const auto& runner = NpuOpRunner(
        "MultinomialWithReplacementD",
        {*x},
        {*out},
        {{"num_samples", num_samples}, {"replacement", replacement}});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    multinomial,
    ops::NPUMultinomialKernel<paddle::platform::NPUDeviceContext, float>,
    ops::NPUMultinomialKernel<paddle::platform::NPUDeviceContext, double>)
#endif
