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

namespace paddle::inference::tensorrt {

/*
 * Gather Op
 */
class GatherOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a gather op to tensorrt gather layer";

    framework::OpDesc op_desc(op, nullptr);
    std::string input_name = op_desc.Input("X").front();
    std::string index_name = op_desc.Input("Index").front();
    std::string output_name = op_desc.Output("Out").front();
    const auto input_tensor = engine_->GetITensor(input_name);
    const auto index_tensor = engine_->GetITensor(index_name);

    int axis = 0;
    if (op_desc.HasAttr("axis")) {
      axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
    }

    auto reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *index_tensor);

    nvinfer1::Dims index_shape{};
    index_shape.nbDims = 1;
    index_shape.d[0] = -1;

    reshape_layer->setReshapeDimensions(index_shape);
    reshape_layer->setName(
        ("Gather: Shuffle: (Output: " + output_name + ")").c_str());

    auto layer = TRT_ENGINE_ADD_LAYER(
        engine_, Gather, *input_tensor, *reshape_layer->getOutput(0), axis);
    layer->setNbElementWiseDims(0);

    ReplenishLayerAndOutput(layer, "gather", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(gather, GatherOpConverter);
