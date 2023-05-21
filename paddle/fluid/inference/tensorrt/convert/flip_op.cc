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
namespace inference {
namespace tensorrt {

class FlipOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
#if IS_TRT_VERSION_GE(8600)
    VLOG(3) << "convert a flip op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);

    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto input_dims = input->getDimensions();

    int rank = input_dims.nbDims;
    int axis = PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("axis"))[0];
    if (axis < 0) {
      axis += rank;
    }

    // expand the batch dimension for dynamic shape
    auto* input_shape = Shape(input);

    auto* batch_dim = Add1DConstantLayer(std::vector<int>{1});
    std::vector<nvinfer1::ITensor*> concat_inputs{batch_dim, input_shape};
    auto* new_shape_tensor = Concat(concat_inputs);

    auto* reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    reshape_layer->setInput(1, *new_shape_tensor);

    auto* reshape_input = reshape_layer->getOutput(0);

    // get the sequence length at the axis dimension
    auto* sequence_lens = GetEleTensorOfShape(input_shape, axis);

    std::cout << "axis <<<" << axis << std::endl;
    auto* reverse_layer = TRT_ENGINE_ADD_LAYER(
        engine_, ReverseSequence, *reshape_input, *sequence_lens);

    reverse_layer->setBatchAxis(0);
    reverse_layer->setSequenceAxis(axis + 1);

    auto* reshape_layer2 =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *reverse_layer->getOutput(0));
    reshape_layer2->setInput(1, *input_shape);

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(reshape_layer2, "flip", {output_name}, test_mode);
#else
    VLOG(3) << "Flip is not supported when TensorRT < 8.6";
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(flip, FlipOpConverter);
