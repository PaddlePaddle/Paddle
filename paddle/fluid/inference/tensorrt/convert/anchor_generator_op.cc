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
#include "paddle/fluid/inference/tensorrt/plugin/anchor_generator_op_plugin.h"

namespace paddle::inference::tensorrt {

/* Anchor Generator Op */
class AnchorGeneratorOpConverter : public OpConverter {
 public:
  void operator()(const paddle::framework::proto::OpDesc& op,
                  const paddle::framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a anchor generator op to tensorrt plugin";
    framework::OpDesc op_desc(op, nullptr);
    std::string input_name = op_desc.Input("Input").front();
    std::string anchor_name = op_desc.Output("Anchors").front();
    std::string variance_name = op_desc.Output("Variances").front();

    auto* input = engine_->GetITensor(input_name);
    const auto input_dims = input->getDimensions();  // C, H, W
    std::vector<std::string> output_names{anchor_name, variance_name};

    const auto anchor_sizes =
        PADDLE_GET_CONST(std::vector<float>, op_desc.GetAttr("anchor_sizes"));
    const auto aspect_ratios =
        PADDLE_GET_CONST(std::vector<float>, op_desc.GetAttr("aspect_ratios"));
    const auto stride =
        PADDLE_GET_CONST(std::vector<float>, op_desc.GetAttr("stride"));
    const auto variances =
        PADDLE_GET_CONST(std::vector<float>, op_desc.GetAttr("variances"));
    const auto offset = PADDLE_GET_CONST(float, op_desc.GetAttr("offset"));
    const int num_anchors = aspect_ratios.size() * anchor_sizes.size();
    bool is_dynamic = true;
    const auto height = input_dims.d[1];
    const auto width = input_dims.d[2];
    const int box_num = width * height * num_anchors;
    const nvinfer1::DataType data_type = nvinfer1::DataType::kFLOAT;

    nvinfer1::IPluginV2* anchor_generator_plugin = nullptr;
    if (is_dynamic) {
      anchor_generator_plugin =
          new plugin::AnchorGeneratorPluginDynamic(data_type,
                                                   anchor_sizes,
                                                   aspect_ratios,
                                                   stride,
                                                   variances,
                                                   offset,
                                                   num_anchors);
    } else {
      anchor_generator_plugin = new plugin::AnchorGeneratorPlugin(data_type,
                                                                  anchor_sizes,
                                                                  aspect_ratios,
                                                                  stride,
                                                                  variances,
                                                                  offset,
                                                                  height,
                                                                  width,
                                                                  num_anchors,
                                                                  box_num);
    }

    std::vector<nvinfer1::ITensor*> anchor_generator_inputs{input};
    auto* anchor_generator_layer =
        engine_->network()->addPluginV2(anchor_generator_inputs.data(),
                                        anchor_generator_inputs.size(),
                                        *anchor_generator_plugin);

    ReplenishLayerAndOutput(
        anchor_generator_layer, "anchor_generator", output_names, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(anchor_generator, AnchorGeneratorOpConverter);
