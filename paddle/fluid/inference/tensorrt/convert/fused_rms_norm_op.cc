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

class FusedRmsNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert fused fused_rms_norm op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);

    // Declare inputs
    auto* x = engine_->GetITensor(op_desc.Input("x")[0]);
    auto* scale = engine_->GetITensor(op_desc.Input("scale")[0]);
    std::vector<nvinfer1::ITensor*> inputs;
    inputs.push_back(x);
    inputs.push_back(scale);

    float epsilon = PADDLE_GET_CONST(float, op_desc.GetAttr("epsilon"));

    nvinfer1::ILayer* layer = nullptr;

    int type = static_cast<int>(nvinfer1::DataType::kHALF);

    auto creator = GetPluginRegistry()->getPluginCreator("Rmsnorm", "1");
    PADDLE_ENFORCE_NE(creator,
                      nullptr,
                      platform::errors::InvalidArgument(
                          "fail to get creator of Rmsnorm Plugin"));
    const std::vector<nvinfer1::PluginField> fields{
        {"eps", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1},
        {"type_id", &type, nvinfer1::PluginFieldType::kINT32, 1}};
    nvinfer1::PluginFieldCollection* pluginPtr =
        static_cast<nvinfer1::PluginFieldCollection*>(
            malloc(sizeof(*pluginPtr) +
                   fields.size() * sizeof(nvinfer1::PluginField)));
    pluginPtr->nbFields = static_cast<int>(fields.size());
    pluginPtr->fields = fields.data();

    auto pluginObj = creator->createPlugin("RmsnormPlugin", pluginPtr);
    auto plugin_layer = engine_->network()->addPluginV2(
        inputs.data(), inputs.size(), *pluginObj);

    layer = plugin_layer;

    std::vector<std::string> output_names;
    output_names.push_back(op_desc.Output("y")[0]);
    RreplenishLayerAndOutput(
        layer, "fused_rms_norm", {output_names}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fused_rms_norm, FusedRmsNormOpConverter);
