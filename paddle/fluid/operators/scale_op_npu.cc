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

template <typename T>
static inline T GetAttrFromTensor(const phi::DenseTensor* tensor) {
  const auto* tensor_data = tensor->data<T>();
  phi::DenseTensor cpu_tensor;
  if (platform::is_gpu_place(tensor->place()) ||
      platform::is_npu_place(tensor->place())) {
    paddle::framework::TensorCopySync(
        *tensor, platform::CPUPlace(), &cpu_tensor);
    tensor_data = cpu_tensor.data<T>();
  }
  return tensor_data[0];
}

template <typename T>
class ScaleNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
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
      auto* scale_tensor = ctx.Input<phi::DenseTensor>("ScaleTensor");
      scale = static_cast<float>(GetAttrFromTensor<T>(scale_tensor));
    }
    if (isinf(scale)) {
      if (signbit(scale)) {
        scale = -std::numeric_limits<float>::max();
      } else {
        scale = std::numeric_limits<float>::max();
      }
    }
    if (!bias_after_scale) {
      bias *= scale;
    }
    out->mutable_data<T>(ctx.GetPlace());

    framework::NPUAttributeMap attrs = {
        {"power", power}, {"scale", scale}, {"shift", bias}};
    const auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    auto op_func = [](const std::vector<Tensor>& inputs,
                      const std::vector<Tensor>& outputs,
                      const NPUAttributeMap& attrs,
                      const platform::NPUDeviceContext& dev_ctx) {
      const auto& muls_runner = NpuOpRunner(
          "Muls", {inputs[0]}, {outputs[0]}, {{"value", attrs.at("scale")}});
      muls_runner.Run(dev_ctx.stream());

      const auto& adds_runner = NpuOpRunner(
          "Adds", {outputs[0]}, {outputs[0]}, {{"value", attrs.at("shift")}});
      adds_runner.Run(dev_ctx.stream());
    };

    if (framework::TransToProtoVarType(x->dtype()) ==
        framework::proto::VarType::INT32) {
      NpuOpRunner::TypeAdapter({*x},
                               {*out},
                               attrs,
                               dev_ctx,
                               op_func,
                               {framework::proto::VarType::INT32},
                               {framework::proto::VarType::INT32});
    } else if (framework::TransToProtoVarType(x->dtype()) ==
               framework::proto::VarType::INT64) {
      NpuOpRunner::TypeAdapter({*x},
                               {*out},
                               attrs,
                               dev_ctx,
                               op_func,
                               {framework::proto::VarType::INT32},
                               {framework::proto::VarType::INT32});
    } else {
      const auto& runner = NpuOpRunner("Power", {*x}, {*out}, attrs);
      runner.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_NPU_KERNEL(
    scale,
    paddle::operators::ScaleNPUKernel<float>,
    paddle::operators::ScaleNPUKernel<paddle::platform::float16>,
    paddle::operators::ScaleNPUKernel<int64_t>,
    paddle::operators::ScaleNPUKernel<int>);
