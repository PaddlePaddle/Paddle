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
 * ConcatOp
 */
class ConcatOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a concat op to tensorrt concat layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    std::vector<nvinfer1::ITensor*> itensors;
    for (auto& input_name : op_desc.Input("X")) {
      itensors.push_back(engine_->GetITensor(input_name));
    }
    int axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
    if (axis < 0) {
      axis = engine_->GetITensor(op_desc.Input("X").front())
                 ->getDimensions()
                 .nbDims +
             axis;
    }
    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Concatenation, itensors.data(), itensors.size());
    layer->setAxis(axis);
    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "concat", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(concat, ConcatOpConverter);
