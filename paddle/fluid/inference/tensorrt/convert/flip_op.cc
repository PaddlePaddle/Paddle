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

class FlipOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a flip op to tensorrt layer";
    std::cout << "convert a flip op to tensorrt layer" << std::endl;
    framework::OpDesc op_desc(op, nullptr);

    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto input_dims = input->getDimensions();
    int rank = input_dims.nbDims;
    int axis =  PADDLE_GET_CONST(int64_t, op_desc.GetAttr("axis"))
    if (axis < 0) axis += rank;
    auto* input_shape = Shape(input);
    auto dims = input_shape->getDimensions()
    auto* sequenceLens = Add1DConstantLayer(dims.d[axis]);
    auto output_name = op_desc.Output("Out")[0];
    auto* reverse_layer = TRT_ENGINE_ADD_LAYER(engine_, ReverseSequence, *input, *sequenceLens);
//    reverse_layer->setBatchAxis(axis);
//    reverse_layer->setSequenceAxis(axis);
    RreplenishLayerAndOutput(reverse_layer,
                             "flip",
                             {output_name + "_value", output_name},
                             test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(flip, FlipOpConverter);
