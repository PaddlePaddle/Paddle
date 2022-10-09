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
#include "paddle/fluid/inference/tensorrt/plugin/lookup_table.h"

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

class FusedLookupTablesOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    if (!engine_->with_dynamic_shape()) {
      PADDLE_THROW(
          platform::errors::Fatal("lookup_table_op must with dynamic shape"));
    }

    framework::OpDesc op_desc(op, nullptr);
    auto ids_name = op_desc.Input("Ids").front();
    auto w_name = op_desc.Input("W").front();
    auto output_name = op_desc.Output("Out").front();
    bool enable_int8 = op_desc.HasAttr("enable_int8");
    std::vector<nvinfer1::ITensor*> plugin_inputs;

    auto ids_dims = engine_->GetITensor(ids_name)->getDimensions();
    if (ids_dims.d[ids_dims.nbDims - 1] == 1) {
      nvinfer1::Dims new_ids_dims;
      new_ids_dims.nbDims = ids_dims.nbDims - 1;
      for (int i = 0; i < ids_dims.nbDims - 1; i++) {
        new_ids_dims.d[i] = 0;
      }
      auto* reshape_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Shuffle, *(engine_->GetITensor(ids_name)));
      reshape_layer->setReshapeDimensions(new_ids_dims);
      reshape_layer->setName(
          ("lookup_table: Shuffle (Output: " + output_name + ")").c_str());
      plugin_inputs.push_back(reshape_layer->getOutput(0));
    } else {
      plugin_inputs.push_back(engine_->GetITensor(ids_name));
    }

    TensorRTEngine::Weight weight;
    auto* w_var = scope.FindVar(w_name);
    auto* w_tensor = w_var->GetMutable<framework::LoDTensor>();
    auto w_dims = w_tensor->dims();
    weight = engine_->GetTrtWeight(w_name, *w_tensor);
    auto weight_size = phi::product(w_dims);
    bool output_fp16;
    if (engine_->precision() == AnalysisConfig::Precision::kFloat32) {
      output_fp16 = false;
    } else {
      output_fp16 = true;
    }

    int32_t weight_width = static_cast<int32_t>(w_dims[1]);

    std::vector<nvinfer1::PluginField> fields;
    fields.emplace_back("lookup_table_weight",
                        weight.get().values,
                        GetPluginFieldType(weight.get().type),
                        static_cast<int32_t>(weight_size));
    fields.emplace_back("lookup_table_weight_width",
                        &weight_width,
                        nvinfer1::PluginFieldType::kINT32,
                        1);
    fields.emplace_back(
        "output_fp16", &output_fp16, nvinfer1::PluginFieldType::kINT32, 1);
    nvinfer1::PluginFieldCollection* plugin_ptr =
        static_cast<nvinfer1::PluginFieldCollection*>(
            malloc(sizeof(*plugin_ptr) +
                   fields.size() * sizeof(nvinfer1::PluginField)));
    plugin_ptr->nbFields = static_cast<int>(fields.size());
    plugin_ptr->fields = fields.data();
    auto creator =
        GetPluginRegistry()->getPluginCreator("LookupTablePluginDynamic", "1");
    auto plugin_obj =
        creator->createPlugin("LookupTablePluginDynamic", plugin_ptr);

    auto plugin_layer = engine_->network()->addPluginV2(
        plugin_inputs.data(), plugin_inputs.size(), *plugin_obj);

    plugin_layer->setName(
        ("lookup_table: (Output: " + output_name + ")").c_str());
    engine_->SetITensor(output_name, plugin_layer->getOutput(0));
    free(plugin_ptr);
    if (enable_int8) {
      float out_scale =
          PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
      engine_->SetTensorDynamicRange(plugin_layer->getOutput(0), out_scale);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(lookup_table, FusedLookupTablesOpConverter);
