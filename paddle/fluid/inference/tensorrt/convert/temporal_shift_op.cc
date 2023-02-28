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
 * TemporalShiftOp.
 */
class TemporalShiftOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid transpose op to tensorrt tranpose layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    const float shift_ratio = PADDLE_GET_CONST(float, op_desc.GetAttr("shift_ratio"));
    const int T = PADDLE_GET_CONST(int, op_desc.GetAttr("seg_num"));

    const auto& input_dims = input->getDimensions();
    int NT = input_dims.d[0];
    int C = input_dims.d[1];
    int H = input_dims.d[2];
    int W = input_dims.d[3];
    int N = NT / T;

    // Reshape input to [N,T,C,H,W]
    auto reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    reshape_layer->setReshapeDimensions(nvinfer1::Dims5(N, T, C, H, W));
    input = reshape_layer->getOutput(0);

    // Pad input
    auto* pad_layer = TRT_ENGINE_ADD_LAYER(engine_, Padding, *input, nvinfer1::Dims4(0, 0, 1, 1), nvinfer1::Dims4(0, 0, 1, 1));
    input = pad_layer->getOutput(0);

    // Slice input
    int slice_size = static_cast<int>(C * shift_ratio);
    auto slice1_layer = TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, nvinfer1::Dims3(0, 0, 0), nvinfer1::Dims3(N, T, slice_size), nvinfer1::Dims3(1, 1, 1));
    auto slice2_layer = TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, nvinfer1::Dims3(0, 2, slice_size), nvinfer1::Dims3(N, T, slice_size), nvinfer1::Dims3(1, 1, 1));
    auto slice3_layer = TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, nvinfer1::Dims3(0, 1, slice_size * 2), nvinfer1::Dims3(N, T, C - slice_size * 2), nvinfer1::Dims3(1, 1, 1));

    // Concatenate slices along the third dimension (C)
    auto concat_layer = TRT_ENGINE_ADD_LAYER(engine_, Concatenation, &slice1_layer->getOutput(0), 3);
    concat_layer->setInput(1, slice2_layer->getOutput(0));
    concat_layer->setInput(2, slice3_layer->getOutput(0));
    concat_layer->setAxis(2);

    // Reshape output to [N*T,C,H,W]
    auto reshape_layer2 = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *concat_layer->getOutput(0));
    reshape_layer2->setReshapeDimensions(nvinfer1::Dims4(NT, C, H, W));

    // Set output
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(reshape_layer2, "temporal_shift", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(temporal_shift, TemporalShiftOp);
