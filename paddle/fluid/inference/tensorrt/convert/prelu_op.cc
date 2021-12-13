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
#include "paddle/fluid/inference/tensorrt/plugin/prelu_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * PRelu converter from fluid to tensorRT.
 */
class PReluOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert fluid prelu op to tensorrt prelu layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    size_t input_num = op_desc.Input("X").size();
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    // Get attrs
    std::string mode = BOOST_GET_CONST(std::string, op_desc.GetAttr("mode"));
    std::string data_format = "NCHW";
    if (op_desc.HasAttr("data_format")) {
      data_format =
          BOOST_GET_CONST(std::string, op_desc.GetAttr("data_format"));
    }
    auto* alpha_var = scope.FindVar(op_desc.Input("Alpha")[0]);
    auto* alpha_tensor = alpha_var->GetMutable<framework::LoDTensor>();

    platform::CPUPlace cpu_place;
    std::unique_ptr<framework::LoDTensor> alpha_tensor_temp(
        new framework::LoDTensor());
    alpha_tensor_temp->Resize(alpha_tensor->dims());
    TensorCopySync(*alpha_tensor, cpu_place, alpha_tensor_temp.get());
    float* alpha_data = alpha_tensor_temp->mutable_data<float>(cpu_place);

    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      plugin::PReluPluginDynamic* plugin = new plugin::PReluPluginDynamic(
          alpha_data, alpha_tensor_temp->numel(), mode, data_format);
      layer = engine_->AddDynamicPlugin(&input, input_num, plugin);
    } else {
#if IS_TRT_VERSION_GE(7000)
      float* alpha_weight_data = engine_->GetWeightCPUData(
          op_desc.Input("Alpha")[0], alpha_tensor, false);
      TensorRTEngine::Weight alpha_weight{
          nvinfer1::DataType::kFLOAT, static_cast<void*>(alpha_weight_data),
          static_cast<size_t>(alpha_tensor->numel())};

      nvinfer1::Dims dims;
      dims.nbDims = 0;
      // jump batch dim
      for (int i = 1; i < alpha_tensor->dims().size(); i++) {
        dims.d[dims.nbDims++] = alpha_tensor->dims()[i];
      }
      for (; dims.nbDims < input->getDimensions().nbDims; dims.nbDims++) {
        dims.d[dims.nbDims] = 1;
      }

      auto alpha_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Constant, dims, alpha_weight.get());
      auto alpha_layer_output = alpha_layer->getOutput(0);

      layer = TRT_ENGINE_ADD_LAYER(engine_, ParametricReLU, *input,
                                   *alpha_layer_output);
#else
      plugin::PReluPlugin* plugin = new plugin::PReluPlugin(
          alpha_data, alpha_tensor_temp->numel(), mode, data_format);
      layer = engine_->AddPlugin(&input, input_num, plugin);
#endif
    }
    // keep alpha tensor to avoid release it's memory
    engine_->SetWeights(op_desc.Input("Alpha")[0],
                        std::move(alpha_tensor_temp));

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "prelu", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(prelu, PReluOpConverter);
