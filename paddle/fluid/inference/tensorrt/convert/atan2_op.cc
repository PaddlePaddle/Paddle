/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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
/*
 * Atan2 Op
 */
class Atan2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "Convert a atan2 op to TensorRT Layer";
    framework::OpDesc op_desc(op, nullptr);
    auto* X1 = engine_->GetITensor(op_desc.Input("X1").front());
    auto* X2 = engine_->GetITensor(op_desc.Input("X2").front());
    // atan2(X1/X2)=arctan(X1/X2)-(X2<0)*{2*(X1<0)-1}*PI

    // arctan(X1/X2)
    // X1/X2
    auto* div_layer = TRT_ENGINE_ADD_LAYER(
        engine_, ElementWise, *X1, *X2, nvinfer1::ElementWiseOperation::kDIV);
    auto* arctan_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                              Unary,
                                              *div_layer->getOutput(0),
                                              nvinfer1::UnaryOperation::kATAN);
    // make constant Tensor
    std::vector<float> zero_vec{0.f};
    std::vector<float> one_vec{1.f};
    std::vector<float> two_vec{2.f};
    std::vector<float> pi_vec{3.1415926535};
    auto zero_tensor = Add1DConstantLayer(zero_vec);
    auto one_tensor = Add1DConstantLayer(one_vec);
    auto two_tensor = Add1DConstantLayer(two_vec);
    auto pi_tensor = Add1DConstantLayer(pi_vec);

    // X2<0
    auto x2_less_layer =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *X2,
                             *BroadcastTensors(X2, zero_tensor),
                             nvinfer1::ElementWiseOperation::kLESS);
    auto x2 = Cast(x2_less_layer->getOutput(0), nvinfer1::DataType::kFLOAT);

    // [2*(X1<0)-1]*PI
    // x1<0
    auto x1_greater_layer =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *X1,
                             *BroadcastTensors(X1, zero_tensor),
                             nvinfer1::ElementWiseOperation::kLESS);
    auto x1 = Cast(x1_greater_layer->getOutput(0), nvinfer1::DataType::kFLOAT);
    x1 = Prod(BroadcastTensors(X1, two_tensor), x1);
    x1 = Sub(x1, BroadcastTensors(X1, one_tensor));
    x1 = Prod(x1, BroadcastTensors(X1, pi_tensor));

    // other=x2*x1
    auto other = Prod(x2, x1);
    // atan2 -> arctan(X1/X2)+y
    auto layer = TRT_ENGINE_ADD_LAYER(engine_,
                                      ElementWise,
                                      *arctan_layer->getOutput(0),
                                      *other,
                                      nvinfer1::ElementWiseOperation::kSUB);
    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "atan2", {output_name}, test_mode);
  }
};
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
REGISTER_TRT_OP_CONVERTER(atan2, Atan2OpConverter);
