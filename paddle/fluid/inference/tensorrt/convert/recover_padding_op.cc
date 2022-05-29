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

/*
 * Recover padding of transformer'input.
 */
class RecoverPadding : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "Recover padding of transformer'output: VarSeqlen -> Padding.";
    if (!engine_->with_dynamic_shape()) {
      PADDLE_THROW(platform::errors::Fatal(
          "recover_padding_op: If you want to use transformer, must "
          "be with dynamic shape"));
    }

    framework::OpDesc op_desc(op, nullptr);
    /*
    auto x_var_name = op_desc.Input(InputNames()).front();
    auto* x_var_desc = block->FindVar(x_var_name);
    const auto x_shape = x_var_desc->GetShape();
    */
    auto input_name = op_desc.Input("Input").front();

    std::cout << "input_name: " << input_name << std::endl;

    std::vector<nvinfer1::ITensor*> plugin_inputs;
    plugin_inputs.push_back(engine_->GetITensor(input_name));
    plugin_inputs.push_back(engine_->GetITensor("pos_id"));
    plugin_inputs.push_back(engine_->GetITensor("mask_id"));
    int input_num = 3;
    auto output_name = op_desc.Output("Out").front();

    plugin::RecoverPaddingPlugin* plugin = new plugin::RecoverPaddingPlugin();
    nvinfer1::ILayer* layer =
        engine_->AddDynamicPlugin(plugin_inputs.data(), input_num, plugin);

    RreplenishLayerAndOutput(layer, "recover_padding", {output_name},
                             test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(recover_padding, RecoverPadding);
