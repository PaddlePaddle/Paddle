/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

class ArgMinOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a arg_min op to tensorrt topk layer";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto input_dims = input->getDimensions();
    int rank = input_dims.nbDims;
    int axis = op_desc.HasAttr("axis")
                   ? PADDLE_GET_CONST(int64_t, op_desc.GetAttr("axis"))
                   : -1;
    if (axis < 0) axis += rank;
    auto* topk_layer = TRT_ENGINE_ADD_LAYER(
        engine_, TopK, *input, nvinfer1::TopKOperation::kMIN, 1, 1 << axis);

    auto output_name = op_desc.Output("Out")[0];
    bool keepdims = PADDLE_GET_CONST(bool, op_desc.GetAttr("keepdims"));
    if (keepdims) {
      ReplenishLayerAndOutput(topk_layer,
                              "arg_min",
                              {output_name + "_value", output_name},
                              test_mode);
    } else {
      auto squeeze_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *topk_layer->getOutput(1));
      auto dims = input_dims;
      dims.nbDims -= 1;
      for (int i = axis; i < dims.nbDims; i++) {
        dims.d[i] = dims.d[i + 1];
      }
      squeeze_layer->setReshapeDimensions(dims);
      ReplenishLayerAndOutput(
          squeeze_layer, "arg_min", {output_name}, test_mode);
    }
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(arg_min, ArgMinOpConverter);
