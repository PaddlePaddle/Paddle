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
 * Gather Op
 */
class GatherOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid gather op to tensorrt gather layer";

    framework::OpDesc op_desc(op, nullptr);
    std::string input_name = op_desc.Input("X").front();
    std::string index_name = op_desc.Input("Index").front();
    std::string output_name = op_desc.Output("Out").front();

    const auto input_tensor = engine_->GetITensor(input_name);
    const auto index_tensor = engine_->GetITensor(index_name);

    const int axis = 0;

    auto layer = TRT_ENGINE_ADD_LAYER(engine_, Gather, *input_tensor,
                                      *index_tensor, axis);

    auto odim = layer->getOutput(0)->getDimensions();

    auto reshape_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *layer->getOutput(0));

    nvinfer1::Dims target_shape{};
    target_shape.nbDims = odim.nbDims - 1;
    for (int i = 0; i < axis; ++i) {
      target_shape.d[i] = odim.d[i];
    }
    target_shape.d[axis] = 0;
    for (int i = axis + 1; i < target_shape.nbDims; ++i) {
      target_shape.d[i] = odim.d[i + 1];
    }

    reshape_layer->setReshapeDimensions(target_shape);

    RreplenishLayerAndOutput(reshape_layer, "gather", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(gather, GatherOpConverter);
