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

#include <NvInferRuntimeCommon.h>
#include <cstddef>
#include <iostream>
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle::inference::tensorrt {

class BitwiseNotConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert bitwise_not op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    nvinfer1::ILayer* layer = nullptr;

    auto* input_tensor = engine_->GetITensor(op_desc.Input("X")[0]);
    nvinfer1::DataType data_type = input_tensor->getType();

    // for bool type: use UnaryOperation::kNOT, for int type: !x = -x - 1
    if (data_type == nvinfer1::DataType::kBOOL) {
      layer = TRT_ENGINE_ADD_LAYER(
          engine_, Unary, *input_tensor, nvinfer1::UnaryOperation::kNOT);
    } else {
      nvinfer1::Dims input_dims = input_tensor->getDimensions();

      // set up a elementwise -1 tensor, can not get the dims info for
      // dynamic_shape so just let it broadcast
      nvinfer1::Dims neg_one_tensor_dims;
      neg_one_tensor_dims.nbDims = input_dims.nbDims;
      for (int i = 0; i < input_dims.nbDims; ++i) {
        neg_one_tensor_dims.d[i] = 1;
      }

      nvinfer1::Weights weights{nvinfer1::DataType::kINT32, new int(-1), 1};
      auto neg_one_tensor =
          TRT_ENGINE_ADD_LAYER(engine_, Constant, neg_one_tensor_dims, weights)
              ->getOutput(0);

      auto mul_neg_one =
          TRT_ENGINE_ADD_LAYER(engine_,
                               ElementWise,
                               *input_tensor,
                               *neg_one_tensor,
                               nvinfer1::ElementWiseOperation::kPROD);

      layer = TRT_ENGINE_ADD_LAYER(engine_,
                                   ElementWise,
                                   *(mul_neg_one->getOutput(0)),
                                   *neg_one_tensor,
                                   nvinfer1::ElementWiseOperation::kSUM);
    }

    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "bitwise_not", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(bitwise_not, BitwiseNotConverter);
