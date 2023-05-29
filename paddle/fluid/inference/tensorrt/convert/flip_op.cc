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
    VLOG(3) << "convert a flip op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);

    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto input_dims = input->getDimensions();
    int rank = input_dims.nbDims;
    std::cout << "<<< rank:" << rank << std::endl;

    auto* input_shape = Shape(input);
    std::vector<int> axis_ =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("axis"));
    std::cout << "<<< axis: " << axis_.back() << std::endl;
    nvinfer1::Dims dummy;
    dummy.nbDims = rank;
    nvinfer1::ITensor* sizes{};
    nvinfer1::ITensor* steps{};
    nvinfer1::ITensor* starts{};

    // for (int i = 0; i < (int)axis_.size() - 1; i++) {
    //   int axis = axis_[i];
    //   if (axis < 0) {
    //     axis += rank;
    //   }

    //   // use Slice to filp one dimension
    //   sizes = input_shape;
    //   std::vector<int> steps_vector{rank, 1};
    //   steps_vector[axis] = -1;
    //   steps = Add1DConstantLayer(steps_vector);

    //   std::vector<int> starts_vector{rank, 0};
    //   starts_vector[axis] = 1;
    //   auto* starts = Add1DConstantLayer(starts_vector);
    //   starts = Prod(input_shape, starts);
    //   starts = Sum(starts, steps);

    //   auto* layer =
    //       TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, dummy, dummy, dummy);

    //   layer->setInput(1, *starts);
    //   layer->setInput(2, *sizes);
    //   layer->setInput(2, *steps);

    //   input = layer->getOutput(0);
    // }

    // use Slice to filp one dimension
    sizes = input_shape;
    std::vector<int> steps_vector(rank, 1);
    steps_vector[axis_.back()] = -1;
    steps = Add1DConstantLayer(steps_vector);

    std::vector<int> starts_vector(rank, 0);
    starts_vector[axis_.back()] = 1;
    starts = Add1DConstantLayer(starts_vector);
    starts = Prod(input_shape, starts);
    starts = Sum(starts, steps);

    auto* slice_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, dummy, dummy, dummy);
    slice_layer->setInput(1, *starts);
    slice_layer->setInput(2, *sizes);
    slice_layer->setInput(3, *steps);

    std::cout << "<<<" << slice_layer->getOutput(0)->getDimensions().nbDims
              << std::endl;
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(slice_layer, "flip", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(flip, FlipOpConverter);
