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

/*
 * PadOp.
 */
class PadOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert pad op to tensorrt IPaddingLayer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    const std::vector<int> paddings =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));

    int pad_size = static_cast<int>(paddings.size());

    nvinfer1::DimsHW pre_pad(paddings[pad_size - 4], paddings[pad_size - 2]);
    nvinfer1::DimsHW post_pad(paddings[pad_size - 3], paddings[pad_size - 1]);

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                       PaddingNd,
                                       *const_cast<nvinfer1::ITensor*>(input),
                                       pre_pad,
                                       post_pad);

    PADDLE_ENFORCE_NOT_NULL(
        layer,
        common::errors::External("add padding layer to tensorrt engine error"));
    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "pad", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(pad, PadOpConverter);
