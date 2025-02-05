/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
 * ClipOp
 */
class ClipOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a clip op to tensorrt layer.";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    float min = PADDLE_GET_CONST(float, op_desc.GetAttr("min"));
    float max = PADDLE_GET_CONST(float, op_desc.GetAttr("max"));
    int32_t rank = input->getDimensions().nbDims;
    nvinfer1::ITensor* input_shape_tensor = Shape(input);
    nvinfer1::DataType data_type = input->getType();
    nvinfer1::ITensor* alphaT{nullptr};
    nvinfer1::ITensor* betaT{nullptr};
    if (data_type == nvinfer1::DataType::kINT32) {
      alphaT =
          FillConstantLayer(input_shape_tensor, rank, static_cast<int>(min));
      betaT =
          FillConstantLayer(input_shape_tensor, rank, static_cast<int>(max));
    } else {
      alphaT = FillConstantLayer(input_shape_tensor, rank, min);
      betaT = FillConstantLayer(input_shape_tensor, rank, max);
    }

    auto* lowerClip = Max(input, alphaT);
    auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                       ElementWise,
                                       *lowerClip,
                                       *betaT,
                                       nvinfer1::ElementWiseOperation::kMIN);

    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "clip", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(clip, ClipOpConverter);
