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
#include "paddle/fluid/inference/tensorrt/plugin/emb_eltwise_layernorm_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class EmbEltwiseLayerNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  const AttachInfo& info) override {
    VLOG(4) << "convert fluid swish op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    std::vector<nvinfer1::ITensor*> inputs;
    auto* word_id = engine_->GetITensor(op_desc.Input("WordId")[0]);
    auto* pos_id = engine_->GetITensor(op_desc.Input("PosId")[0]);
    auto* sent_id = engine_->GetITensor(op_desc.Input("SentId")[0]);

    inputs.push_back(word_id);
    inputs.push_back(pos_id);
    inputs.push_back(sent_id);

    auto get_persistable_data = [&](const std::string& arg_name,
                                    framework::DDim* dims) -> float* {
      std::string var_name = op_desc.Input(arg_name).front();
      auto* temp_var = scope.FindVar(var_name);
      auto* temp_tensor = temp_var->GetMutable<framework::LoDTensor>();
      (*dims) = temp_tensor->dims();

      auto* temp_data = engine_->GetWeightCPUData(var_name, temp_tensor, false);
      return temp_data;
    };

    framework::DDim word_emb_dims, pos_emb_dims, sent_emb_dims, bias_dims,
        scale_dims;

    nvinfer1::DataType input_type = nvinfer1::DataType::kFLOAT;
    if (engine_->WithFp16()) {
      input_type = nvinfer1::DataType::kHALF;
    }
    auto* word_emb = get_persistable_data("WordEmb", &word_emb_dims);
    auto* pos_emb = get_persistable_data("PosEmb", &pos_emb_dims);
    auto* sent_emb = get_persistable_data("SentEmb", &sent_emb_dims);
    auto* bias = get_persistable_data("Bias", &bias_dims);
    auto* scale = get_persistable_data("Scale", &scale_dims);
    int64_t word_emb_size = framework::product(word_emb_dims);
    int64_t pos_emb_size = framework::product(pos_emb_dims);
    int64_t sent_emb_size = framework::product(sent_emb_dims);
    int64_t bias_size = framework::product(bias_dims);
    int64_t scale_size = framework::product(scale_dims);
    int hidden = word_emb_dims[1];
    float eps = boost::get<float>(op_desc.GetAttr("epsilon"));

    nvinfer1::ILayer* layer = nullptr;
    plugin::EmbEltwiseLayernormPluginDynamic* plugin =
        new plugin::EmbEltwiseLayernormPluginDynamic(
            word_emb, pos_emb, sent_emb, bias, scale, word_emb_size,
            pos_emb_size, sent_emb_size, bias_size, scale_size, hidden, eps,
            input_type);
    layer = engine_->AddPluginV2(inputs.data(), 3, plugin);
    // Get output
    size_t output_num = op_desc.Output("Out").size();
    PADDLE_ENFORCE(output_num == 1);

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "emb_eltwise_layernorm", {output_name},
                             info.test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fused_embedding_eltwise_layernorm,
                          EmbEltwiseLayerNormOpConverter);
