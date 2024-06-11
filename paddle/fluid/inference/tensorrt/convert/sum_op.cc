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

namespace paddle::inference::tensorrt {

class SumOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a sum op to tensorrt sum layer";

    framework::OpDesc op_desc(op, nullptr);
    nvinfer1::ILayer* layer = nullptr;
    // Declare the first input
    auto* sum_tmp = engine_->GetITensor(op_desc.Input("X")[0]);
    if (op_desc.Input("X").size() == 1) {
      layer = TRT_ENGINE_ADD_LAYER(engine_, Identity, *sum_tmp);
    } else {
      for (size_t i = 1; i < op_desc.Input("X").size(); i++) {
        auto* input_i = engine_->GetITensor(op_desc.Input("X")[i]);
        layer = TRT_ENGINE_ADD_LAYER(engine_,
                                     ElementWise,
                                     *input_i,
                                     *sum_tmp,
                                     nvinfer1::ElementWiseOperation::kSUM);
        sum_tmp = layer->getOutput(0);
      }
    }
    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "sum", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(sum, SumOpConverter);
