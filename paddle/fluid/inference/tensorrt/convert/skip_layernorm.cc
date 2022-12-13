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
#include "paddle/fluid/inference/tensorrt/convert/utils.h"
#include "paddle/fluid/inference/tensorrt/engine.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class SkipLayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert fused skip layernorm op to tensorrt layer";
    PADDLE_ENFORCE_EQ(engine_->with_dynamic_shape(),
                      true,
                      platform::errors::InvalidArgument(
                          "Skip_layernorm must run the dynamic shape mode."));
    framework::OpDesc op_desc(op, nullptr);
    auto GetWeight =
        [&](const std::string& arg_name) -> TensorRTEngine::Weight {
      std::string var_name = op_desc.Input(arg_name).front();
      auto* temp_var = scope.FindVar(var_name);
      auto* temp_tensor = temp_var->GetMutable<phi::DenseTensor>();
      auto weight = engine_->GetTrtWeight(var_name, *temp_tensor);
      return weight;
    };
    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);
    auto* input2 = engine_->GetITensor(op_desc.Input("Y")[0]);
    std::vector<nvinfer1::ITensor*> inputs;
    inputs.push_back(input1);
    inputs.push_back(input2);

    bool enable_int8 = false;
    if (op_desc.HasAttr("enable_int8")) {
      enable_int8 = PADDLE_GET_CONST(bool, op_desc.GetAttr("enable_int8"));
    }
    auto bias_weight = GetWeight("Bias").get();
    auto scale_weight = GetWeight("Scale").get();
    nvinfer1::ILayer* layer = nullptr;
    bool flag_varseqlen = engine_->use_varseqlen() &&
                          engine_->tensorrt_transformer_posid() != "" &&
                          engine_->tensorrt_transformer_maskid() != "";
    if (flag_varseqlen && engine_->with_interleaved()) {
      VLOG(4) << "fused skip_layernorm op: use_varseqlen and with_interleaved";
      if (!enable_int8) {
        PADDLE_THROW(
            platform::errors::Fatal("use with_interleaved must be int8."));
      }
      auto creator = GetPluginRegistry()->getPluginCreator(
          "CustomSkipLayerNormPluginDynamic", "3");
      PADDLE_ENFORCE_NE(
          creator,
          nullptr,
          platform::errors::InvalidArgument(
              "fail to get creator of CustomSkipLayerNormPluginDynamic"));
      const std::vector<nvinfer1::PluginField> fields{
          {"beta",
           bias_weight.values,
           GetPluginFieldType(bias_weight.type),
           static_cast<int32_t>(bias_weight.count)},
          {"gamma",
           scale_weight.values,
           GetPluginFieldType(scale_weight.type),
           static_cast<int32_t>(scale_weight.count)}};
      nvinfer1::PluginFieldCollection* pluginPtr =
          static_cast<nvinfer1::PluginFieldCollection*>(
              malloc(sizeof(*pluginPtr) +
                     fields.size() * sizeof(nvinfer1::PluginField)));
      pluginPtr->nbFields = static_cast<int32_t>(fields.size());
      pluginPtr->fields = fields.data();

      auto pluginObj =
          creator->createPlugin("CustomSkipLayerNormPluginDynamic", pluginPtr);
      auto plugin_layer = engine_->network()->addPluginV2(
          inputs.data(), inputs.size(), *pluginObj);

      PADDLE_ENFORCE_NE(
          plugin_layer,
          nullptr,
          platform::errors::InvalidArgument(
              "fail to add CustomSkipLayerNormPluginDynamic layer"));
      layer = plugin_layer;
    } else {
      auto creator = GetPluginRegistry()->getPluginCreator(
          "CustomSkipLayerNormPluginDynamic", "2");
      PADDLE_ENFORCE_NE(
          creator,
          nullptr,
          platform::errors::InvalidArgument(
              "fail to get creator of CustomSkipLayerNormPluginDynamic"));
      int32_t type = static_cast<int32_t>((engine_->WithFp16() == 1)
                                              ? nvinfer1::DataType::kHALF
                                              : nvinfer1::DataType::kFLOAT);
      if (enable_int8) {
        type = static_cast<int32_t>(nvinfer1::DataType::kHALF);
      }
      int32_t ld = PADDLE_GET_CONST(int32_t, op_desc.GetAttr("ld"));
      PADDLE_ENFORCE_GT(ld,
                        0,
                        platform::errors::InvalidArgument(
                            "in CustomSkipLayerNormPluginDynamic hidden "
                            "dimension should > 0"));

      const std::vector<nvinfer1::PluginField> fields{
          {"type_id", &type, nvinfer1::PluginFieldType::kINT32, 1},
          {"ld", &ld, nvinfer1::PluginFieldType::kINT32, 1},
          {"beta",
           bias_weight.values,
           GetPluginFieldType(bias_weight.type),
           static_cast<int32_t>(bias_weight.count)},
          {"gamma",
           scale_weight.values,
           GetPluginFieldType(scale_weight.type),
           static_cast<int32_t>(scale_weight.count)},
      };
      nvinfer1::PluginFieldCollection* pluginPtr =
          static_cast<nvinfer1::PluginFieldCollection*>(
              malloc(sizeof(*pluginPtr) +
                     fields.size() *
                         sizeof(nvinfer1::PluginField)));  // remember to free
      pluginPtr->nbFields = static_cast<int32_t>(fields.size());
      pluginPtr->fields = fields.data();

      auto pluginObj =
          creator->createPlugin("CustomSkipLayerNormPluginDynamic", pluginPtr);
      auto plugin_layer = engine_->network()->addPluginV2(
          inputs.data(), inputs.size(), *pluginObj);

      PADDLE_ENFORCE_NE(
          plugin_layer,
          nullptr,
          platform::errors::InvalidArgument(
              "fail to add CustomSkipLayerNormPluginDynamic layer"));
      layer = plugin_layer;
    }
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "skip_layernorm", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(skip_layernorm, SkipLayerNormOpConverter);
