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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class TrilTriuNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    int diagonal = ctx.Attr<int>("diagonal");
    bool lower = ctx.Attr<bool>("lower");

    out->mutable_data<T>(ctx.GetPlace());

    std::string op_type = lower ? "Tril" : "Triu";

    framework::NPUAttributeMap attr_input = {{"diagonal", diagonal}};

    const auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();

    auto op_func_tril = [](const std::vector<Tensor>& inputs,
                           const std::vector<Tensor>& outputs,
                           const NPUAttributeMap& attrs,
                           const platform::NPUDeviceContext& dev_ctx) {
      const auto& runner = NpuOpRunner("Tril", inputs, outputs, attrs);
      runner.Run(dev_ctx.stream());
    };

    auto op_func_triu = [](const std::vector<Tensor>& inputs,
                           const std::vector<Tensor>& outputs,
                           const NPUAttributeMap& attrs,
                           const platform::NPUDeviceContext& dev_ctx) {
      const auto& runner = NpuOpRunner("Triu", inputs, outputs, attrs);
      runner.Run(dev_ctx.stream());
    };

    if (framework::TransToProtoVarType(x->dtype()) ==
        framework::proto::VarType::BOOL) {
      if (lower) {
        NpuOpRunner::TypeAdapter({*x},
                                 {*out},
                                 attr_input,
                                 dev_ctx,
                                 op_func_tril,
                                 {framework::proto::VarType::UINT8},
                                 {framework::proto::VarType::UINT8});
      } else {
        NpuOpRunner::TypeAdapter({*x},
                                 {*out},
                                 attr_input,
                                 dev_ctx,
                                 op_func_triu,
                                 {framework::proto::VarType::UINT8},
                                 {framework::proto::VarType::UINT8});
      }
    } else {
      const auto& runner = NpuOpRunner(op_type, {*x}, {*out}, attr_input);
      runner.Run(dev_ctx.stream());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    tril_triu,
    ops::TrilTriuNPUKernel<plat::NPUDeviceContext, float>,
    ops::TrilTriuNPUKernel<plat::NPUDeviceContext, int>,
    ops::TrilTriuNPUKernel<plat::NPUDeviceContext, bool>,
    ops::TrilTriuNPUKernel<plat::NPUDeviceContext, plat::float16>);
