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
 * TransposeOp
 */
class Transpose2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid transpose2 op to tensorrt transpose layer "
               "without bias";

    framework::OpDesc op_desc(op, nullptr);
    auto input_tensor = engine_->GetITensor(op_desc.Input("X").front());

    auto axis = BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("axis"));

    nvinfer1::Permutation permutation{};

    for (size_t i = 1; i < axis.size(); ++i) {
      permutation.order[i - 1] = axis[i] - 1;
    }

    auto* transpose_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input_tensor);
    transpose_layer->setFirstTranspose(permutation);

    auto output_name = op_desc.Output("Out").front();
    RreplenishLayerAndOutput(transpose_layer, "transpose2", {output_name},
                             test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(transpose2, Transpose2OpConverter);
