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
 * SoftMaxOp, ISoftMaxLayer in TRT. This Layer doesn't has weights.
 */
class SoftMaxOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid softmax op to tensorrt softmax layer";

    nvinfer1::ILayer* layer = nullptr;
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);
    nvinfer1::Dims dims_input = input1->getDimensions();
    int num_ele = 1;
    for (int i = 0; i < dims_input.nbDims; i++) {
      num_ele *= dims_input.d[i];
    }

    bool need_do_reshape = (num_ele != dims_input.d[0]);
    if (need_do_reshape) {
      nvinfer1::DimsHW reshape_dims(dims_input.d[0], num_ele / dims_input.d[0]);
      auto* reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input1);
      reshape_layer->setReshapeDimensions(reshape_dims);
      layer = reshape_layer;
      input1 = layer->getOutput(0);
    }

    auto* softmax_layer = TRT_ENGINE_ADD_LAYER(engine_, SoftMax, *input1);

    layer = softmax_layer;
    if (need_do_reshape) {
      softmax_layer->setAxes(2);
      auto* reshape_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *layer->getOutput(0));
      reshape_layer->setReshapeDimensions(dims_input);
      layer = reshape_layer;
    }

    auto output_name = op_desc.Output("Out")[0];
    engine_->SetITensor(output_name, layer->getOutput(0));
    if (test_mode) {
      engine_->DeclareOutput(output_name);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(softmax);
REGISTER_TRT_OP_CONVERTER(softmax, SoftMaxOpConverter);
