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

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class PrelnSkipLayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
#if IS_TRT_VERSION_GE(7000)
    VLOG(4) << "convert fused preln_skip_layernorm op to tensorrt layer";
    if (!(engine_->use_oss() && engine_->with_interleaved())) {
      PADDLE_THROW(platform::errors::Fatal(
          "PrelnErnie: If you want to use oss, must be with interleaved"));
    }
    framework::OpDesc op_desc(op, nullptr);
    bool enable_int8 = op_desc.HasAttr("enable_int8");
    if (!enable_int8) {
      PADDLE_THROW(
          platform::errors::Fatal("use with_interleaved must be int8."));
    }
    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);
    auto* input2 = engine_->GetITensor(op_desc.Input("Y")[0]);
    std::vector<nvinfer1::ITensor*> inputs;
    inputs.push_back(input1);
    inputs.push_back(input2);

    auto get_persistable_data = [&](const std::string& arg_name,
                                    framework::DDim* dims) -> float* {
      std::string var_name = op_desc.Input(arg_name).front();
      auto* temp_var = scope.FindVar(var_name);
      auto* temp_tensor = temp_var->GetMutable<framework::LoDTensor>();
      (*dims) = temp_tensor->dims();

      auto* temp_data = engine_->GetWeightCPUData(var_name, temp_tensor, false);
      return temp_data;
    };

    framework::DDim bias_dims, scale_dims;
    auto* bias = get_persistable_data("Bias", &bias_dims);
    auto* scale = get_persistable_data("Scale", &scale_dims);
    int bias_size = phi::product(bias_dims);
    int scale_size = phi::product(scale_dims);

    nvinfer1::ILayer* layer = nullptr;

    VLOG(4) << "fused preln_skip_layernorm op: use_oss and with_interleaved";

    auto creator = GetPluginRegistry()->getPluginCreator(
        "CustomSkipLayerNormPluginDynamic", "4");
    PADDLE_ENFORCE_NE(
        creator, nullptr,
        platform::errors::InvalidArgument(
            "fail to get creator of CustomPrelnSkipLayerNormPluginDynamic"));
    const std::vector<nvinfer1::PluginField> fields{
        {"beta", bias, nvinfer1::PluginFieldType::kFLOAT32, bias_size},
        { "gamma",
          scale,
          nvinfer1::PluginFieldType::kFLOAT32,
          scale_size }};
    nvinfer1::PluginFieldCollection* pluginPtr =
        static_cast<nvinfer1::PluginFieldCollection*>(
            malloc(sizeof(*pluginPtr) +
                   fields.size() * sizeof(nvinfer1::PluginField)));
    pluginPtr->nbFields = static_cast<int>(fields.size());
    pluginPtr->fields = fields.data();

    auto pluginObj =
        creator->createPlugin("CustomSkipLayerNormPluginDynamic", pluginPtr);
    auto plugin_layer = engine_->network()->addPluginV2(
        inputs.data(), inputs.size(), *pluginObj);

    PADDLE_ENFORCE_NE(
        plugin_layer, nullptr,
        platform::errors::InvalidArgument(
            "fail to add CustomPrelnSkipLayerNormPluginDynamic layer"));
    layer = plugin_layer;

    std::vector<std::string> output_names;
    output_names.push_back(op_desc.Output("Out_0")[0]);
    output_names.push_back(op_desc.Output("Out_1")[0]);
    RreplenishLayerAndOutput(layer, "preln_skip_layernorm", {output_names},
                             test_mode);
#else
    PADDLE_THROW(platform::errors::Fatal(
        "PreInErnie want to use oss, must be with interleaved, "
        "your TRT version is no less than 7.0"));
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(preln_skip_layernorm, PrelnSkipLayerNormOpConverter);
