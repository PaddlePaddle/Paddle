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

    auto input_dims = input->getDimensions();

    const int NT = input_dims.d[0];
    const int C = input_dims.d[1];
    const int H = input_dims.d[2];
    const int W = input_dims.d[3];
    const int N = NT / T;

    // Reshape input to [N,C,H,W,T]
    auto reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    nvinfer1::Dims reshape_dims{5, {N, C, H, W, T}};
    reshape_layer->setReshapeDimensions(reshape_dims);

    // Pad input
    auto* pad_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                           Padding,
                                           *reshape_layer->getOutput(0),
                                           nvinfer1::DimsHW{0, 1},
                                           nvinfer1::DimsHW{0, 1});

    // Reshape input to [N,T,C,H,W]
    auto reshape_layer2 = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *pad_layer->getOutput(0));
    nvinfer1::Dims reshape_dims2{5, {N, T + 2, C, H, W}};
    reshape_layer2->setReshapeDimensions(reshape_dims2);

    // print pad_layer->getOutput(0)->getDimensions()
    auto pad_dims = pad_layer->getOutput(0)->getDimensions();
    int dims = pad_dims.nbDims;
    for (int i = 0; i < dims; ++i) {
        std::cout << pad_dims.d[i] << " ";
    }
    std::cout << std::endl;

    // Slice input
//    int slice_c = int(C * shift_ratio);
//    int slice_c2 = int(C * shift_ratio * 2);
//
//    auto* slice1_layer = TRT_ENGINE_ADD_LAYER(engine_,
//                                              Slice,
//                                              *pad_layer->getOutput(0),
//                                              nvinfer1::Dims{5, {0, 0, 0, 0, 0}},
//                                              nvinfer1::Dims{5, {N, slice_c, H, W, T}},
//                                              nvinfer1::Dims{5, {1, 1, 1, 1, 1}});
//    auto* slice2_layer = TRT_ENGINE_ADD_LAYER(engine_,
//                                              Slice,
//                                              *pad_layer->getOutput(0),
//                                              nvinfer1::Dims{5, {0, slice_c, 0, 0, 2}},
//                                              nvinfer1::Dims{5, {N, slice_c, H, W, T}},
//                                              nvinfer1::Dims{5, {1, 1, 1, 1, 1}});
//    auto* slice3_layer = TRT_ENGINE_ADD_LAYER(engine_,
//                                              Slice,
//                                              *pad_layer->getOutput(0),
//                                              nvinfer1::Dims{5, {0, slice_c2, 0, 0, 1}},
//                                              nvinfer1::Dims{5, {N, C - slice_c2, H, W, T}},
//                                              nvinfer1::Dims{5, {1, 1, 1, 1, 1}});
    int slice_c = int(C * shift_ratio);
    int slice_c2 = int(C * shift_ratio * 2);
    auto* slice1_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                              Slice,
                                              *reshape_layer2->getOutput(0),
                                              nvinfer1::Dims{5, {0, 0, 0, 0, 0}},
                                              nvinfer1::Dims{5, {N, T, slice_c, H, W}},
                                              nvinfer1::Dims{5, {1, 1, 1, 1, 1}});
    auto* slice2_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                              Slice,
                                              *reshape_layer2->getOutput(0),
                                              nvinfer1::Dims{5, {0, 2, slice_c, 0, 0}},
                                              nvinfer1::Dims{5, {N, T, slice_c, H, W}},
                                              nvinfer1::Dims{5, {1, 1, 1, 1, 1}});
    auto* slice3_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                              Slice,
                                              *reshape_layer2->getOutput(0),
                                              nvinfer1::Dims{5, {0, 1, slice_c2, 0, 0}},
                                              nvinfer1::Dims{5, {N, T, C - slice_c2, H, W}},
                                              nvinfer1::Dims{5, {1, 1, 1, 1, 1}});

    // Concatenate slices along the third dimension (C)
    nvinfer1::IConcatenationLayer* concat_layer;
    if(!slice_c){
        nvinfer1::ITensor* concat_inputs[2] = {slice2_layer->getOutput(0),
                                               slice3_layer->getOutput(0)};
        concat_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                              Concatenation,
                                              concat_inputs, 2);
        concat_layer->setAxis(2);
    }
    else{
        nvinfer1::ITensor* concat_inputs[3] = {slice1_layer->getOutput(0),
                                               slice2_layer->getOutput(0),
                                               slice3_layer->getOutput(0)};
        concat_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                                  Concatenation,
                                                  concat_inputs, 3);
        concat_layer->setAxis(2);
    }

    // Reshape output to [N*T,C,H,W]
    nvinfer1::Dims output_shape{4, {N * T, C, H, W}};
    auto* reshape_layer3 = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *concat_layer->getOutput(0));
    reshape_layer3->setReshapeDimensions(output_shape);

    // Set output
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(reshape_layer3, "temporal_shift", {output_name}, test_mode);

  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(temporal_shift, TemporalShiftOpConverter);
