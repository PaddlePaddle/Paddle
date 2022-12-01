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

namespace nvinfer1 {
class ILayer;
}  // namespace nvinfer1
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

class WhereIndexOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(0) << "convert fluid where_index op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    int input_num = op_desc.Input("Condition").size();
    PADDLE_ENFORCE_EQ(
        input_num,
        1,
        platform::errors::InvalidArgument(
            "The input Condition's size must equal to 1 in TRT nonzero op."
            " But received Condition's size %d.",
            input_num));
    auto* input = engine_->GetITensor(op_desc.Input("Condition")[0]);
    // Get output
    size_t output_num = op_desc.Output("Out").size();
    PADDLE_ENFORCE_EQ(
        output_num,
        1UL,
        platform::errors::InvalidArgument(
            "The output Out's size must equal to 1 in TRT nonzero op. "
            "But received Out's size %u.",
            output_num));

    nvinfer1::INonZeroLayer* nonzero_layer =
        TRT_ENGINE_ADD_LAYER(engine_, NonZero, *input);

    std::vector<int> axis = {1, 0};
    nvinfer1::Permutation perm;
    for (int i = 0; i < 2; i++) {
      perm.order[i] = axis[i];
    }
    auto* transpose_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *nonzero_layer->getOutput(0));
    transpose_layer->setFirstTranspose(perm);
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(
        transpose_layer, "where_index", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(where_index, WhereIndexOpConverter);
