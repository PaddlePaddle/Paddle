/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/instance_norm_op_plugin.h"

namespace paddle::inference::tensorrt {

class InstanceNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert fluid prelu op to tensorrt instance norm layer";

    framework::OpDesc op_desc(op, nullptr);
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    float eps = PADDLE_GET_CONST(float, op_desc.GetAttr("epsilon"));

    auto* scale_var = scope.FindVar(op_desc.Input("Scale")[0]);
    auto* bias_var = scope.FindVar(op_desc.Input("Bias")[0]);
    PADDLE_ENFORCE_NOT_NULL(
        scale_var,
        common::errors::InvalidArgument(
            "Input [Scale] of instance_norm op converter should not be null"));
    PADDLE_ENFORCE_NOT_NULL(
        bias_var,
        common::errors::InvalidArgument(
            "Input [Bias] of instance_norm op converter should not be null"));
    auto* scale_tensor = scale_var->GetMutable<phi::DenseTensor>();
    auto* bias_tensor = bias_var->GetMutable<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(
        scale_tensor->numel(),
        bias_tensor->numel(),
        common::errors::InvalidArgument(
            "Num of input [Scale] and [Bias] of instance_norm op converter "
            "should be equal. Got Scale num = %ld, but Bias num = %ld",
            scale_tensor->numel(),
            bias_tensor->numel()));
    auto* scale_d = scale_tensor->data<float>();
    auto* bias_d = bias_tensor->data<float>();

    std::vector<float> scale_v;
    std::vector<float> bias_v;
    for (int i = 0; i < scale_tensor->numel(); i++) {
      scale_v.push_back(scale_d[i]);
      bias_v.push_back(bias_d[i]);
    }

    nvinfer1::IPluginV2* plugin = nullptr;
    plugin = new plugin::InstanceNormPluginDynamic(eps, scale_v, bias_v);

    std::vector<nvinfer1::ITensor*> instance_norm_inputs{input};
    auto* layer = engine_->network()->addPluginV2(
        instance_norm_inputs.data(), instance_norm_inputs.size(), *plugin);

    auto output_name = op_desc.Output("Y")[0];
    ReplenishLayerAndOutput(layer, "instance_norm", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(instance_norm, InstanceNormOpConverter);
