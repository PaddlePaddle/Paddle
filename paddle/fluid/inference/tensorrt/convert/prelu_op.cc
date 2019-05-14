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
    int input_num = op_desc.Input("X").size();
    PADDLE_ENFORCE(input_num == 1);
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    // Get output
    size_t output_num = op_desc.Output("Out").size();
    PADDLE_ENFORCE(output_num == 1);
    // Get attrs
    std::string mode = boost::get<std::string>(op_desc.GetAttr("mode"));
    //
    auto* alpha_var = scope.FindVar(op_desc.Input("Alpha")[0]);
    PADDLE_ENFORCE_NOT_NULL(alpha_var);
    auto* alpha_tensor = alpha_var->GetMutable<framework::LoDTensor>();

    platform::CPUPlace cpu_place;
    std::unique_ptr<framework::LoDTensor> alpha_tensor_temp(
        new framework::LoDTensor());
    alpha_tensor_temp->Resize(alpha_tensor->dims());
    TensorCopySync(*alpha_tensor, cpu_place, alpha_tensor_temp.get());
    float* alpha_data = alpha_tensor_temp->mutable_data<float>(cpu_place);

    plugin::PReluPlugin* plugin =
        new plugin::PReluPlugin(alpha_data, alpha_tensor_temp->numel(), mode);
    nvinfer1::IPluginLayer* layer =
        engine_->AddPlugin(&input, input_num, plugin);
    // keep alpha tensor to avoid release it's memory
    engine_->weight_map[op_desc.Input("Alpha")[0]] =
        std::move(alpha_tensor_temp);

    std::string layer_name = "prelu (Output: ";
    auto output_name = op_desc.Output("Out")[0];
    layer->getOutput(0)->setName(output_name.c_str());
    engine_->SetITensor(output_name, layer->getOutput(0));
    layer_name += output_name;
    if (test_mode) {
      engine_->DeclareOutput(output_name);
    }
    layer->setName((layer_name + ")").c_str());
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(prelu, PReluOpConverter);
