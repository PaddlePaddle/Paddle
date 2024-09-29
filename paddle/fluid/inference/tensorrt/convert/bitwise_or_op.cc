// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <NvInferRuntimeCommon.h>
#include <cstddef>
#include <iostream>
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle::inference::tensorrt {

class BitwiseOrConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert bitwise_or op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    nvinfer1::ILayer* layer = nullptr;

    auto* input_tensor = engine_->GetITensor(op_desc.Input("X")[0]);
    nvinfer1::DataType data_type = input_tensor->getType();

    auto* y_tensor = engine_->GetITensor(op_desc.Input("Y")[0]);

    // for bool type
    if (data_type == nvinfer1::DataType::kBOOL) {
      layer = TRT_ENGINE_ADD_LAYER(engine_,
                                   ElementWise,
                                   *input_tensor,
                                   *y_tensor,
                                   nvinfer1::ElementWiseOperation::kOR);
    } else {
      PADDLE_THROW(common::errors::Fatal(
          "bitwise_or TRT converter is only supported on bool"));
    }

    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "bitwise_or", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(bitwise_or, BitwiseOrConverter);
