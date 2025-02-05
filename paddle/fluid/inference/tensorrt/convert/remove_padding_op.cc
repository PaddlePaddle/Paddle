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
#include "paddle/fluid/inference/tensorrt/plugin/remove_padding_plugin.h"

namespace paddle::inference::tensorrt {

/*
 * Remove padding of transformer'input.
 */
class RemovePadding : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    if (!engine_->with_dynamic_shape()) {
      PADDLE_THROW(common::errors::Fatal(
          "remove_padding_op: If you want to use transformer, must "
          "be with dynamic shape"));
    }

    framework::OpDesc op_desc(op, nullptr);
    auto input_name = op_desc.Input("Input").front();
    auto output_name = op_desc.Output("Out").front();
    std::vector<nvinfer1::ITensor*> plugin_inputs;
    plugin_inputs.push_back(engine_->GetITensor(input_name));
    plugin_inputs.push_back(engine_->GetITensor("pos_id"));
    plugin_inputs.push_back(engine_->GetITensor("word_id"));
    size_t input_num = plugin_inputs.size();
    plugin::RemovePaddingPlugin* plugin = new plugin::RemovePaddingPlugin();
    nvinfer1::ILayer* layer =
        engine_->AddDynamicPlugin(plugin_inputs.data(), input_num, plugin);
    layer->setName(("remove_padding: (Output: " + output_name + ")").c_str());
    if (engine_->with_interleaved()) {
      VLOG(3) << "with_interleaved data format: Remove padding of "
                 "transformer'input: Padding -> VarSeqlen.";
      if (!op_desc.HasAttr("out_threshold")) {
        PADDLE_THROW(
            common::errors::Fatal("use with_interleaved must be int8."));
      }
      float out_scale =
          PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
      engine_->SetTensorDynamicRange(layer->getOutput(0), out_scale);
      auto* transpose =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(layer->getOutput(0)));
      transpose->setSecondTranspose({2, 1, 0, 3});
      transpose->setName(
          ("remove_padding (with_interleaved): transpose(Output: " +
           output_name + ")")
              .c_str());
      engine_->SetITensor(output_name, transpose->getOutput(0));
    } else {
      VLOG(3) << "normal data format: Remove padding of transformer'input: "
                 "Padding -> VarSeqlen.";
      engine_->SetITensor(output_name, layer->getOutput(0));
    }
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(remove_padding, RemovePadding);
