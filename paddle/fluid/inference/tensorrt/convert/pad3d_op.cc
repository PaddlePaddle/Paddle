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
namespace framework {
class Scope;

namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * Pad3dOp.
 */
class Pad3dOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid transpose op to tensorrt tranpose layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    const std::vector<int> paddings =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));

    nvinfer1::Dims pre_pad, post_pad;

    pre_pad.nbDims = 3;
    pre_pad.d[0] = paddings[0];
    pre_pad.d[1] = paddings[2];
    pre_pad.d[2] = paddings[4];

    post_pad.nbDims = 3;
    post_pad.d[0] = paddings[1];
    post_pad.d[1] = paddings[3];
    post_pad.d[2] = paddings[5];

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                       PaddingNd,
                                       *const_cast<nvinfer1::ITensor*>(input),
                                       pre_pad,
                                       post_pad);

    PADDLE_ENFORCE_NOT_NULL(layer,
                            platform::errors::External(
                                "add padding layer to tensorrt engine error"));
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "pad3d", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(pad3d, Pad3dOpConverter);
