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

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/gather_nd_op_plugin.h"

namespace paddle::inference::tensorrt {

class GatherNdOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    auto input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto index = engine_->GetITensor(op_desc.Input("Index")[0]);
    auto output_name = op_desc.Output("Out")[0];

    // AddGatherV2 is supported by the trt version of 8.2.
#if IS_TRT_VERSION_GE(8200)
    VLOG(3) << "convert gather_nd op to tensorrt gather_nd layer";

    auto layer = TRT_ENGINE_ADD_LAYER(
        engine_, GatherV2, *input, *index, nvinfer1::GatherMode::kND);
    layer->setNbElementWiseDims(0);
    ReplenishLayerAndOutput(layer, "gather_nd", {output_name}, test_mode);
#else
    VLOG(4) << "convert a paddle gather_nd op to tensorrt gather_nd plugin";

    // Declare inputs
    std::vector<nvinfer1::ITensor*> inputs;
    inputs.emplace_back(input);
    inputs.emplace_back(index);

    nvinfer1::ILayer* layer = nullptr;
    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
    plugin::GatherNdPluginDynamic* plugin =
        new plugin::GatherNdPluginDynamic(with_fp16);
    layer = engine_->AddDynamicPlugin(inputs.data(), inputs.size(), plugin);

    std::string layer_name = "gather_nd (Output: ";
    layer->getOutput(0)->setName(output_name.c_str());
    engine_->SetITensor(output_name, layer->getOutput(0));
    layer_name += output_name;
    if (test_mode) {
      engine_->DeclareOutput(output_name);
    }
    layer->setName((layer_name + ")").c_str());
#endif
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(gather_nd, GatherNdOpConverter);
