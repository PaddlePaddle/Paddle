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
#include "paddle/fluid/inference/tensorrt/plugin/skip_layernorm_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class SkipLayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
#if IS_TRT_VERSION_GE(6000)
    VLOG(4) << "convert fused skip layernorm op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);
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

      auto* temp_data = engine_->GetWeightCPUData(var_name, temp_tensor);
      return temp_data;
    };

    framework::DDim bias_dims, scale_dims;
    auto* bias = get_persistable_data("Bias", &bias_dims);
    auto* scale = get_persistable_data("Scale", &scale_dims);
    int bias_size = phi::product(bias_dims);
    int scale_size = phi::product(scale_dims);
    bool enable_int8 = op_desc.HasAttr("enable_int8");

    nvinfer1::ILayer* layer = nullptr;

    if (engine_->use_oss()) {
      if (engine_->with_interleaved()) {
        VLOG(4) << "fused skip_layernorm op: use_oss and with_interleaved";
        if (!enable_int8) {
          PADDLE_THROW(
              platform::errors::Fatal("use with_interleaved must be int8."));
        }
        auto creator = GetPluginRegistry()->getPluginCreator(
            "CustomSkipLayerNormPluginDynamic", "3");
        PADDLE_ENFORCE_NE(
            creator, nullptr,
            platform::errors::InvalidArgument(
                "fail to get creator of CustomSkipLayerNormPluginDynamic"));
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

        auto pluginObj = creator->createPlugin(
            "CustomSkipLayerNormPluginDynamic", pluginPtr);
        auto plugin_layer = engine_->network()->addPluginV2(
            inputs.data(), inputs.size(), *pluginObj);

        PADDLE_ENFORCE_NE(
            plugin_layer, nullptr,
            platform::errors::InvalidArgument(
                "fail to add CustomSkipLayerNormPluginDynamic layer"));
        layer = plugin_layer;
      } else {
        auto creator = GetPluginRegistry()->getPluginCreator(
            "CustomSkipLayerNormPluginDynamic", "2");
        PADDLE_ENFORCE_NE(
            creator, nullptr,
            platform::errors::InvalidArgument(
                "fail to get creator of CustomSkipLayerNormPluginDynamic"));
        int type = static_cast<int>((engine_->WithFp16() == 1)
                                        ? nvinfer1::DataType::kHALF
                                        : nvinfer1::DataType::kFLOAT);
        int ld = input1->getDimensions().d[2];  // hidden dimension
        PADDLE_ENFORCE_GT(ld, 0,
                          platform::errors::InvalidArgument(
                              "in CustomSkipLayerNormPluginDynamic hidden "
                              "dimension should > 0"));
        if (enable_int8) {
          type = static_cast<int>(nvinfer1::DataType::kHALF);
        }

        const std::vector<nvinfer1::PluginField> fields{
            {"type_id", &type, nvinfer1::PluginFieldType::kINT32, 1},
            {"ld", &ld, nvinfer1::PluginFieldType::kINT32, 1},
            {"beta", bias, nvinfer1::PluginFieldType::kFLOAT32, bias_size},
            {"gamma", scale, nvinfer1::PluginFieldType::kFLOAT32, scale_size},
        };
        nvinfer1::PluginFieldCollection* pluginPtr =
            static_cast<nvinfer1::PluginFieldCollection*>(
                malloc(sizeof(*pluginPtr) +
                       fields.size() *
                           sizeof(nvinfer1::PluginField)));  // remember to free
        pluginPtr->nbFields = static_cast<int>(fields.size());
        pluginPtr->fields = fields.data();

        auto pluginObj = creator->createPlugin(
            "CustomSkipLayerNormPluginDynamic", pluginPtr);
        auto plugin_layer = engine_->network()->addPluginV2(
            inputs.data(), inputs.size(), *pluginObj);

        PADDLE_ENFORCE_NE(
            plugin_layer, nullptr,
            platform::errors::InvalidArgument(
                "fail to add CustomSkipLayerNormPluginDynamic layer"));
        layer = plugin_layer;
      }
    } else {
      float eps = BOOST_GET_CONST(float, op_desc.GetAttr("epsilon"));
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::SkipLayerNormPluginDynamic* plugin =
          new plugin::SkipLayerNormPluginDynamic(bias, scale, bias_size,
                                                 scale_size, eps, with_fp16);
      layer = engine_->AddDynamicPlugin(inputs.data(), 2, plugin);
    }

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "skip_layernorm", {output_name}, test_mode);
#else
    PADDLE_THROW(platform::errors::Fatal(
        "You are running the TRT Dynamic Shape mode, need to confirm that "
        "your TRT version is no less than 6.0"));
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(skip_layernorm, SkipLayerNormOpConverter);
