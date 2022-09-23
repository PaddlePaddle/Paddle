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
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/preln_residual_bias_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

using half = paddle::platform::float16;
class PrelnResidualBiasOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert fused preln_residual_bias op to tensorrt layer";
    if (!engine_->with_dynamic_shape()) {
      PADDLE_THROW(platform::errors::Fatal(
          "Unsupported static mode. Please set dynamic shape of inputs."));
    }
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
      auto* temp_data = const_cast<float*>(static_cast<const float*>(
          engine_->GetFp32TrtWeight(var_name, *temp_tensor).get().values));
      return temp_data;
    };
    framework::DDim bias_dims, scale_dims, ele_bias_dims;
    auto* bias = get_persistable_data("Bias", &bias_dims);
    auto* scale = get_persistable_data("Scale", &scale_dims);
    auto const& vars = op_desc.Inputs(false);
    bool has_bias = vars.find("EleBias") != vars.end();
    float* ele_bias =
        has_bias ? get_persistable_data("EleBias", &ele_bias_dims) : nullptr;

    int bias_size = phi::product(bias_dims);

    int scale_size = phi::product(scale_dims);
    int ele_bias_size = has_bias ? phi::product(ele_bias_dims) : 0;
    float epsilon = PADDLE_GET_CONST(float, op_desc.GetAttr("epsilon"));
    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
    if (engine_->precision() == AnalysisConfig::Precision::kInt8) {
      with_fp16 = true;
    }

    nvinfer1::ILayer* layer = nullptr;
    plugin::DynamicPluginTensorRT* plugin = nullptr;
    if (with_fp16) {
      half* half_ele_bias_data = nullptr;
      if (ele_bias_size > 0) {
        half_ele_bias_data = new half[ele_bias_size];
        for (int i = 0; i < ele_bias_size; i++) {
          half_ele_bias_data[i] = static_cast<half>(ele_bias[i]);
        }
      }
      plugin = new plugin::PrelnResidualBiasPluginDynamic(
          bias,
          scale,
          ele_bias_size > 0 ? half_ele_bias_data : nullptr,
          bias_size,
          scale_size,
          ele_bias_size,
          epsilon,
          with_fp16);
    } else {
      plugin = new plugin::PrelnResidualBiasPluginDynamic(bias,
                                                          scale,
                                                          ele_bias,
                                                          bias_size,
                                                          scale_size,
                                                          ele_bias_size,
                                                          epsilon,
                                                          with_fp16);
    }

    std::vector<nvinfer1::ITensor*> plugin_inputs;
    plugin_inputs.emplace_back(input1);
    plugin_inputs.emplace_back(input2);
    layer = engine_->AddDynamicPlugin(plugin_inputs.data(), 2, plugin);
    std::vector<std::string> output_names;
    output_names.push_back(op_desc.Output("Out_0")[0]);
    output_names.push_back(op_desc.Output("Out_1")[0]);
    RreplenishLayerAndOutput(
        layer, "preln_residual_bias", output_names, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(preln_residual_bias, PrelnResidualBiasOpConverter);
