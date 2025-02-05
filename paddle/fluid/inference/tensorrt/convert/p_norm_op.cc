/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
 * p_norm Op
 */
class PNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a p_norm op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    std::string input_name = op_desc.Input("X").front();
    std::string output_name = op_desc.Output("Out").front();
    auto* input_tensor = engine_->GetITensor(input_name);
    int rank = input_tensor->getDimensions().nbDims;
    int axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
    bool keepdim = PADDLE_GET_CONST(bool, op_desc.GetAttr("keepdim"));
    if (axis < 0) {
      axis += rank;
    }
    uint32_t axisMask = 1 << axis;
    auto* prod_tensor = Prod(input_tensor, input_tensor);
    auto* prod_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                            Reduce,
                                            *prod_tensor,
                                            nvinfer1::ReduceOperation::kSUM,
                                            axisMask,
                                            keepdim);
    auto* reduce_tensor = prod_layer->getOutput(0);
    auto* sqrt_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Unary, *reduce_tensor, nvinfer1::UnaryOperation::kSQRT);
    ReplenishLayerAndOutput(sqrt_layer, "p_norm", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(p_norm, PNormOpConverter);
