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

class LinspaceOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    // #if IS_TRT_VERSION_GE(7000)
    VLOG(4) << "convert a linspace op to tensorrt linspace layer";

    framework::OpDesc op_desc(op, nullptr);
    auto* input_num = engine_->GetITensor(op_desc.Input("Num")[0]);
    auto* input_start = engine_->GetITensor(op_desc.Input("Start")[0]);
    auto* input_stop = engine_->GetITensor(op_desc.Input("Stop")[0]);
    int dtype = PADDLE_GET_CONST(int, op_desc.GetAttr("dtype"));
    auto output_name = op_desc.Output("Out")[0];
    nvinfer1::ITensor* beta_tensor;

    auto* input_start_tensor =
        TRT_ENGINE_ADD_LAYER(
            engine_, Cast, *input_start, nvinfer1::DataType::kFLOAT)
            ->getOutput(0);
    auto* input_stop_tensor =
        TRT_ENGINE_ADD_LAYER(
            engine_, Cast, *input_stop, nvinfer1::DataType::kFLOAT)
            ->getOutput(0);
    auto* input_num_tensor =
        TRT_ENGINE_ADD_LAYER(
            engine_, Cast, *input_num, nvinfer1::DataType::kFLOAT)
            ->getOutput(0);

    auto* new_num_tensor = Sub(input_num_tensor, Add1DConstantLayer(1.0f));
    auto* dis_tensor = Sub(input_stop_tensor, input_start_tensor);

    beta_tensor = Div(dis_tensor, new_num_tensor);

    nvinfer1::Dims alpha_shape;
    alpha_shape.nbDims = 0;
    alpha_shape.d[0] = 1;
    auto* alpha_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input_start_tensor);
    alpha_layer->setInput(1,
                          *Add1DConstantLayer(1, "alpha_reshape_tensor", true));
    alpha_layer->setReshapeDimensions(alpha_shape);
    auto* alpha_tensor = alpha_layer->getOutput(0);

    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Fill, nvinfer1::Dims{}, nvinfer1::FillOperation::kLINSPACE);
    layer->setInput(0, *input_num);
    layer->setInput(1, *alpha_tensor);
    layer->setInput(2, *beta_tensor);
    if (dtype == 2 || dtype == 3) {
      auto* cast_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Cast, *(layer->getOutput(0)), nvinfer1::DataType::kINT32);
      RreplenishLayerAndOutput(
          cast_layer, "linspace", {output_name}, test_mode);
    } else {
      RreplenishLayerAndOutput(layer, "linspace", {output_name}, test_mode);
    }

    // #endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(linspace, LinspaceOpConverter);
