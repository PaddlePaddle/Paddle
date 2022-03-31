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

#include <vector>
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace framework {
class Scope;
namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

class GroupNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid group_norm op";

    framework::OpDesc op_desc(op, nullptr);

    auto* input_itensor = engine_->GetITensor(op_desc.Input("X").front());

    int groups = BOOST_GET_CONST(int, op_desc.GetAttr("groups"));
    float epsilon = BOOST_GET_CONST(float, op_desc.GetAttr("epsilon"));

    std::string scale_name = op_desc.Input("Scale").front();
    std::string bias_name = op_desc.Input("Bias").front();

    // get the presistable var's data
    auto get_persistable_data = [&](const std::string& var_name,
                                    framework::DDim* dims) -> float* {
      auto* temp_var = scope.FindVar(var_name);
      auto* temp_tensor = temp_var->GetMutable<framework::LoDTensor>();
      (*dims) = temp_tensor->dims();

      auto* temp_data = engine_->GetWeightCPUData(var_name, temp_tensor, false);
      return temp_data;
    };

    framework::DDim scale_dims;
    framework::DDim bias_dims;
    float* scale_data = get_persistable_data(scale_name, &scale_dims);
    float* bias_data = get_persistable_data(bias_name, &bias_dims);

    int64_t scale_numel = phi::product(scale_dims);
    int64_t bias_numel = phi::product(bias_dims);

    TensorRTEngine::Weight scale_weights{nvinfer1::DataType::kFLOAT,
                                         static_cast<void*>(scale_data),
                                         static_cast<size_t>(scale_numel)};
    TensorRTEngine::Weight bias_weights{nvinfer1::DataType::kFLOAT,
                                        static_cast<void*>(bias_data),
                                        static_cast<size_t>(bias_numel)};

    nvinfer1::Dims scale_nv_dims;
    nvinfer1::Dims bias_nv_dims;
    scale_nv_dims.nbDims = scale_dims.size();
    bias_nv_dims.nbDims = bias_dims.size();
    for (int i = 0; i < scale_dims.size(); i++) {
      scale_nv_dims.d[i] = scale_dims.at(i);
    }
    for (int i = 0; i < bias_dims.size(); i++) {
      bias_nv_dims.d[i] = bias_dims.at(i);
    }

    auto* scale_layer = TRT_ENGINE_ADD_LAYER(engine_, Constant, scale_nv_dims,
                                             scale_weights.get());
    auto* bias_layer = TRT_ENGINE_ADD_LAYER(engine_, Constant, bias_nv_dims,
                                            bias_weights.get());

    std::vector<nvinfer1::ITensor*> plugin_inputs;
    plugin_inputs.emplace_back(input_itensor);
    plugin_inputs.emplace_back(scale_layer->getOutput(0));
    plugin_inputs.emplace_back(bias_layer->getOutput(0));

    const std::vector<nvinfer1::PluginField> fields{
        {"eps", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1},
        {"num_groups", &groups, nvinfer1::PluginFieldType::kINT32, 1},
    };

    nvinfer1::PluginFieldCollection* plugin_collections =
        static_cast<nvinfer1::PluginFieldCollection*>(
            malloc(sizeof(*plugin_collections) +
                   fields.size() * sizeof(nvinfer1::PluginField)));
    plugin_collections->nbFields = static_cast<int>(fields.size());
    plugin_collections->fields = fields.data();

    auto creator =
        GetPluginRegistry()->getPluginCreator("GroupNormalizationPlugin", "1");
    auto group_norm_plugin =
        creator->createPlugin("GroupNormalizationPlugin", plugin_collections);
    free(plugin_collections);

    auto group_norm_plugin_layer = engine_->network()->addPluginV2(
        plugin_inputs.data(), plugin_inputs.size(), *group_norm_plugin);

    auto output_name = op_desc.Output("Y")[0];
    RreplenishLayerAndOutput(group_norm_plugin_layer, "group_norm",
                             {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(group_norm, GroupNormOpConverter);
