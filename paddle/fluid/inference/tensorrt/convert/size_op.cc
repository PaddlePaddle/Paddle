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
#include <iostream>
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle::inference::tensorrt {

class SizeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert size op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("Input")[0]);
    nvinfer1::ITensor* input_shape_tensor = Shape(input);
    uint32_t reduce_dim = 1;
    auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                       Reduce,
                                       *input_shape_tensor,
                                       nvinfer1::ReduceOperation::kPROD,
                                       reduce_dim,
                                       false);
    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "size", {output_name}, test_mode);
  }
};
}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(size, SizeOpConverter);
