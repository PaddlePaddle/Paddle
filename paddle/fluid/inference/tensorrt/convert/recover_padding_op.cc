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
#include "paddle/fluid/inference/tensorrt/plugin/recover_padding_plugin.h"

namespace paddle::inference::tensorrt {

/*
 * Recover padding of transformer'input.
 */
class RecoverPadding : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    if (!engine_->with_dynamic_shape()) {
      PADDLE_THROW(common::errors::Fatal(
          "recover_padding_op: If you want to use transformer, must "
          "be with dynamic shape"));
    }
    framework::OpDesc op_desc(op, nullptr);
    auto input_name = op_desc.Input("Input").front();
    auto input = engine_->GetITensor(input_name);
    auto output_name = op_desc.Output("Out").front();
    std::vector<nvinfer1::ITensor*> plugin_inputs;
    if (engine_->with_interleaved()) {
      VLOG(3) << "with_interleaved data format: Recover padding of "
                 "transformer'output: VarSeqlen -> Padding.";
      if (!op_desc.HasAttr("out_threshold")) {
        PADDLE_THROW(
            common::errors::Fatal("use with_interleaved must be int8."));
      }
      auto* transpose = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      transpose->setSecondTranspose({2, 1, 0, 3});
      auto* transpose_output = transpose->getOutput(0);
      transpose->setName(
          ("recover_padding(with_interleaved): transpose(Output: " +
           output_name + ")")
              .c_str());
      float out_scale =
          PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
      engine_->SetTensorDynamicRange(transpose_output, out_scale);
      plugin_inputs.push_back(transpose_output);
    } else {
      VLOG(3) << "normal data format: Recover padding of transformer'output: "
                 "VarSeqlen -> Padding.";
      plugin_inputs.push_back(input);
    }
    plugin_inputs.push_back(engine_->GetITensor("pos_id"));
    plugin_inputs.push_back(engine_->GetITensor("mask_id"));
    size_t input_num = plugin_inputs.size();
    plugin::RecoverPaddingPlugin* plugin = new plugin::RecoverPaddingPlugin();
    nvinfer1::ILayer* layer =
        engine_->AddDynamicPlugin(plugin_inputs.data(), input_num, plugin);
    ReplenishLayerAndOutput(layer, "recover_padding", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(recover_padding, RecoverPadding);
