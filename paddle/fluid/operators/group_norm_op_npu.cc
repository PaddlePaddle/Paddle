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

#include "paddle/fluid/operators/group_norm_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;

template <typename DeviceContext, typename T>
class GroupNormNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    const float epsilon = ctx.Attr<float>("epsilon");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* x = ctx.Input<Tensor>("X");

    auto* y = ctx.Output<Tensor>("Y");
    auto* mean = ctx.Output<Tensor>("Mean");
    auto* var = ctx.Output<Tensor>("Variance");
    const auto G = ctx.Attr<int>("groups");

    const auto x_dims = x->dims();
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);

    Tensor default_scale(x->type());
    if (!scale) {
      default_scale.mutable_data<T>(framework::make_ddim({C}), place);
      Tensor value(x->type());
      value.mutable_data<T>({1}, place);
      FillNpuTensorWithConstant<T>(&value, static_cast<T>(1.0));
      const auto& runner =
          NpuOpRunner("FillD", {value}, {default_scale}, {{"dims", {C}}});
      runner.Run(stream);
      scale = &default_scale;
    }

    Tensor default_bias(x->type());
    if (!bias) {
      default_bias.mutable_data<T>(framework::make_ddim({C}), place);
      Tensor value(x->type());
      value.mutable_data<T>({1}, place);
      FillNpuTensorWithConstant<T>(&value, static_cast<T>(0));
      const auto& runner =
          NpuOpRunner("FillD", {value}, {default_bias}, {{"dims", {C}}});
      runner.Run(stream);
      bias = &default_bias;
    }

    y->mutable_data<T>(place);
    mean->mutable_data<T>(place);
    var->mutable_data<T>(place);

    const auto& runner = NpuOpRunner(
        "GroupNorm", {*x, *scale, *bias}, {*y, *mean, *var},
        {{"epsilon", epsilon},
         {"data_format", (data_layout == DataLayout::kNCHW ? "NCHW" : "NHWC")},
         {"is_training", false},
         {"num_groups", G}});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(
    group_norm,
    ops::GroupNormNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::GroupNormNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::GroupNormNPUKernel<paddle::platform::NPUDeviceContext,
                            paddle::platform::float16>);
