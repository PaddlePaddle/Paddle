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

namespace paddle {
namespace inference {
namespace tensorrt {

class GatherNdOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid gather_nd op to tensorrt gather_nd layer";

    framework::OpDesc op_desc(op, nullptr);
    std::string input_name = op_desc.Input("X").front();
    std::string index_name = op_desc.Input("Index").front();
    std::string output_name = op_desc.Output("Out").front();
    const auto input_tensor = engine_->GetITensor(input_name);
    const auto index_tensor = engine_->GetITensor(index_name);

    auto layer = TRT_ENGINE_ADD_LAYER(engine_,
                                      GatherV2,
                                      *input_tensor,
                                      *index_tensor,
                                      nvinfer1::GatherMode::kND);
    layer->setNbElementWiseDims(0);

    RreplenishLayerAndOutput(layer, "gather_nd", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(gather_nd, GatherNdOpConverter);
