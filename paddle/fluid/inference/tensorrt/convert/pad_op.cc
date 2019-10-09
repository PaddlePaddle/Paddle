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
 * PadOp.
 */
class PadOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid transpose op to tensorrt tranpose layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    const std::vector<int> paddings =
        boost::get<std::vector<int>>(op_desc.GetAttr("paddings"));
    const float pad_value = boost::get<float>(op_desc.GetAttr("pad_value"));

    nvinfer1::Dims input_shape = input->getDimensions();
    int nbDims = input_shape.nbDims;
    int pad_size = static_cast<int>(paddings.size());
    PADDLE_ENFORCE_GE(nbDims, 2);
    PADDLE_ENFORCE_EQ((nbDims + 1) * 2, pad_size);
    PADDLE_ENFORCE(pad_value == 0.0, "The pad layer of TRT only support zero.");

    nvinfer1::DimsHW pre_pad(paddings[pad_size - 4], paddings[pad_size - 2]);
    nvinfer1::DimsHW post_pad(paddings[pad_size - 3], paddings[pad_size - 1]);

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Padding,
                                       *const_cast<nvinfer1::ITensor*>(input),
                                       pre_pad, post_pad);

    PADDLE_ENFORCE(layer != nullptr);
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "pad", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(pad, PadOpConverter);
