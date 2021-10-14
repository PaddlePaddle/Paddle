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

#include "paddle/fluid/operators/scale_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename T>
class ScaleNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto scale = ctx.Attr<float>("scale");
    auto bias = ctx.Attr<float>("bias");
    auto bias_after_scale = ctx.Attr<bool>("bias_after_scale");
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    float power = 1.0;
    VLOG(4) << "scale:" << scale << ", bias:" << bias
            << " ,bias_after_scale:" << bias_after_scale;
    if (ctx.HasInput("ScaleTensor")) {
      auto* scale_tensor = ctx.Input<framework::Tensor>("ScaleTensor");
      scale = static_cast<float>(GetAttrFromTensor<T>(scale_tensor));
    }

    if (!bias_after_scale) {
      bias *= scale;
    }
    out->mutable_data<T>(ctx.GetPlace());
    const auto& runner =
        NpuOpRunner("Power", {*x}, {*out},
                    {{"power", power}, {"scale", scale}, {"shift", bias}});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_NPU_KERNEL(
    scale, paddle::operators::ScaleNPUKernel<float>,
    paddle::operators::ScaleNPUKernel<paddle::platform::float16>);
