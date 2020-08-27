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
namespace inference {
namespace tensorrt {

/*
 * MulOp, IMatrixMultiplyLayer in TRT. This Layer doesn't has weights.
 */
class MulOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid mul op to tensorrt mul layer without bias";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);
    auto* input2 = engine_->GetITensor(op_desc.Input("Y")[0]);

    bool transpose_x = BOOST_GET_CONST(bool, op_desc.GetAttr("transpose_X"));
    bool transpose_y = BOOST_GET_CONST(bool, op_desc.GetAttr("transpose_Y"));

    // float scale = BOOST_GET_CONST(float, op_desc.GetAttr("alpha"));
    // assert(scale == 1.f);

    // Both the input1 and input2 do not need transpose.
    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, MatrixMultiply, *const_cast<nvinfer1::ITensor*>(input1),
        transpose_x, *const_cast<nvinfer1::ITensor*>(input2), transpose_y);

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "matmul", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(matmul, MulOpConverter);
