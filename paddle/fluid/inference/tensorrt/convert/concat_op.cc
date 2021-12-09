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
 * ConcatOp
 */
class ConcatOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a paddle concat op to tensorrt concat layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    std::vector<nvinfer1::ITensor*> itensors;
    for (auto& input_name : op_desc.Input("X")) {
      itensors.push_back(engine_->GetITensor(input_name));
    }
    int axis = BOOST_GET_CONST(int, op_desc.GetAttr("axis"));

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Concatenation, itensors.data(),
                                       itensors.size());
    if (!engine_->with_dynamic_shape()) {
      axis = axis - 1;  // Remove batch dim
    }
    layer->setAxis(axis);
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "concat", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(concat, ConcatOpConverter);
