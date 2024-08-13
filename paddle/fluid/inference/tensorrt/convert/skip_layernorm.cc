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
#include "paddle/phi/common/data_type.h"

namespace paddle::inference::tensorrt {

class SkipLayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert fused skip layernorm op to tensorrt layer";
    PADDLE_ENFORCE_EQ(engine_->with_dynamic_shape(),
                      true,
                      common::errors::InvalidArgument(
                          "Skip_layernorm must run the dynamic shape mode."));
    framework::OpDesc op_desc(op, nullptr);
    auto output_name = op_desc.Output("Out")[0];
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

    bool enable_int8 = (engine_->precision() == phi::DataType::INT8);
    float x_scale = 0;
    float y_scale = 0;

    if (enable_int8) {
      if (op_desc.HasAttr("X")) {
        x_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("X"));
        engine_->SetTensorDynamicRange(input1, x_scale);
      }
      if (op_desc.HasAttr("Y")) {
        y_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("Y"));
        engine_->SetTensorDynamicRange(input2, y_scale);
      }
    }

    nvinfer1::Dims dims_x = input1->getDimensions();
    int32_t x_rank = dims_x.nbDims;
    nvinfer1::Dims dims_y = input2->getDimensions();
    int32_t y_rank = dims_y.nbDims;

    if ((x_rank == 2 && y_rank == 4) || (y_rank == 2 && x_rank == 4)) {
      if (x_rank == 2 && y_rank == 4) {
        auto* reshape_before_skip_layer_n =
            TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input1);
        std::vector<nvinfer1::ITensor*> reshape_before_tensor;
        reshape_before_tensor.push_back(GetEleTensorOfShape(Shape(input1), 0));
        reshape_before_tensor.push_back(GetEleTensorOfShape(Shape(input1), 1));
        reshape_before_tensor.push_back(Add1DConstantLayer(1));
        reshape_before_tensor.push_back(Add1DConstantLayer(1));
        reshape_before_skip_layer_n->setInput(1,
                                              *Concat(reshape_before_tensor));
        reshape_before_skip_layer_n->setName(
            ("reshape_before_skip_layer_n(Output: " + output_name + ")")
                .c_str());
        input1 = reshape_before_skip_layer_n->getOutput(0);

        if (enable_int8) {
          if (op_desc.HasAttr("X")) {
            engine_->SetTensorDynamicRange(input1, x_scale);
          }
        }
      } else {
        auto* reshape_before_skip_layer_n =
            TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input2);
        std::vector<nvinfer1::ITensor*> reshape_before_tensor;
        reshape_before_tensor.push_back(GetEleTensorOfShape(Shape(input2), 0));
        reshape_before_tensor.push_back(GetEleTensorOfShape(Shape(input2), 1));
        reshape_before_tensor.push_back(Add1DConstantLayer(1));
        reshape_before_tensor.push_back(Add1DConstantLayer(1));
        reshape_before_skip_layer_n->setInput(1,
                                              *Concat(reshape_before_tensor));
        reshape_before_skip_layer_n->setName(
            ("reshape_before_skip_layer_n(Output: " + output_name + ")")
                .c_str());
        input2 = reshape_before_skip_layer_n->getOutput(0);

        if (enable_int8) {
          if (op_desc.HasAttr("Y")) {
            engine_->SetTensorDynamicRange(input2, y_scale);
          }
        }
      }
    }

    std::vector<nvinfer1::ITensor*> inputs;
    inputs.push_back(input1);
    inputs.push_back(input2);

    std::vector<float> smooth_scale;
    bool use_smooth = false;
    if (op_desc.HasAttr("smooth_scale")) {
      smooth_scale =
          PADDLE_GET_CONST(std::vector<float>, op_desc.GetAttr("smooth_scale"));
      use_smooth = true;
    }

    auto bias_weight = GetWeight("Bias").get();
    auto scale_weight = GetWeight("Scale").get();
    nvinfer1::ILayer* layer = nullptr;
    bool flag_varseqlen = engine_->use_varseqlen() &&
                          !engine_->tensorrt_transformer_posid().empty() &&
                          !engine_->tensorrt_transformer_maskid().empty();
    if (flag_varseqlen && engine_->with_interleaved()) {
      VLOG(4) << "fused skip_layernorm op: use_varseqlen and with_interleaved";
      if (!enable_int8) {
        PADDLE_THROW(
            common::errors::Fatal("use with_interleaved must be int8."));
      }
      auto creator = GetPluginRegistry()->getPluginCreator(
          "CustomSkipLayerNormPluginDynamic", "3");
      PADDLE_ENFORCE_NE(
          creator,
          nullptr,
          common::errors::InvalidArgument(
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
      std::unique_ptr<nvinfer1::PluginFieldCollection> pluginPtr(
          new nvinfer1::PluginFieldCollection);
      pluginPtr->nbFields = static_cast<int32_t>(fields.size());
      pluginPtr->fields = fields.data();

      auto pluginObj = creator->createPlugin("CustomSkipLayerNormPluginDynamic",
                                             pluginPtr.get());

      pluginPtr.reset();

      auto plugin_layer = engine_->network()->addPluginV2(
          inputs.data(), inputs.size(), *pluginObj);

      PADDLE_ENFORCE_NE(
          plugin_layer,
          nullptr,
          common::errors::InvalidArgument(
              "fail to add CustomSkipLayerNormPluginDynamic layer"));
      layer = plugin_layer;
    } else {
      auto creator = GetPluginRegistry()->getPluginCreator(
          "CustomSkipLayerNormPluginDynamic", "2");
      PADDLE_ENFORCE_NE(
          creator,
          nullptr,
          common::errors::InvalidArgument(
              "fail to get creator of CustomSkipLayerNormPluginDynamic"));
      int32_t type = static_cast<int32_t>((engine_->WithFp16() == 1)
                                              ? nvinfer1::DataType::kHALF
                                              : nvinfer1::DataType::kFLOAT);
      if (enable_int8) {
        type = static_cast<int32_t>(nvinfer1::DataType::kHALF);
      }
      int32_t hidden_size =
          PADDLE_GET_CONST(int32_t, op_desc.GetAttr("hidden_size"));
      PADDLE_ENFORCE_GT(hidden_size,
                        0,
                        common::errors::InvalidArgument(
                            "in CustomSkipLayerNormPluginDynamic hidden "
                            "dimension should > 0"));

      std::vector<nvinfer1::PluginField> fields{
          {"type_id", &type, nvinfer1::PluginFieldType::kINT32, 1},
          {"ld", &hidden_size, nvinfer1::PluginFieldType::kINT32, 1},
          {"beta",
           bias_weight.values,
           GetPluginFieldType(bias_weight.type),
           static_cast<int32_t>(bias_weight.count)},
          {"gamma",
           scale_weight.values,
           GetPluginFieldType(scale_weight.type),
           static_cast<int32_t>(scale_weight.count)},
      };

      if (use_smooth) {
        VLOG(4) << "using special method, make sure you have correct version "
                   "of tensorrt";
        type = static_cast<int32_t>(nvinfer1::DataType::kINT8);
        fields.push_back({"smooth_scale",
                          smooth_scale.data(),
                          nvinfer1::PluginFieldType::kFLOAT32,
                          static_cast<int32_t>(smooth_scale.size())});
        std::unique_ptr<nvinfer1::PluginFieldCollection> pluginPtr(
            new nvinfer1::PluginFieldCollection);
        pluginPtr->nbFields = static_cast<int32_t>(fields.size());
        pluginPtr->fields = fields.data();

        auto pluginObj = creator->createPlugin(
            "CustomSkipLayerNormPluginDynamicWithSmooth", pluginPtr.get());

        pluginPtr.reset();

        auto plugin_layer = engine_->network()->addPluginV2(
            inputs.data(), inputs.size(), *pluginObj);

        PADDLE_ENFORCE_NE(
            plugin_layer,
            nullptr,
            common::errors::InvalidArgument(
                "fail to add CustomSkipLayerNormPluginDynamicWithSmooth "
                "layer"));
        layer = plugin_layer;
      } else {
        std::unique_ptr<nvinfer1::PluginFieldCollection> pluginPtr(
            new nvinfer1::PluginFieldCollection);
        pluginPtr->nbFields = static_cast<int32_t>(fields.size());
        pluginPtr->fields = fields.data();

        auto pluginObj = creator->createPlugin(
            "CustomSkipLayerNormPluginDynamic", pluginPtr.get());

        pluginPtr.reset();

        auto plugin_layer = engine_->network()->addPluginV2(
            inputs.data(), inputs.size(), *pluginObj);

        PADDLE_ENFORCE_NE(
            plugin_layer,
            nullptr,
            common::errors::InvalidArgument(
                "fail to add CustomSkipLayerNormPluginDynamic layer"));
        layer = plugin_layer;
      }
    }
    ReplenishLayerAndOutput(layer, "skip_layernorm", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(skip_layernorm, SkipLayerNormOpConverter);
