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
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert pad op to tensorrt IPaddingLayer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    nvinfer1::ITensor* transpose_layer1_out = nullptr;
    const std::vector<int> base_paddings =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
    std::vector<int> paddings({base_paddings.begin(), base_paddings.end()});
    // auto in_dims = input->getDimensions();
    bool flag_nchw = true;
    if (base_paddings.size() == 8) {
      for (int i = 0; i < 4; ++i) {
        if (paddings[i] != 0) {
          flag_nchw = false;
          break;
        }
      }
      // transpose to nchw
      if (!flag_nchw) {
        paddings[6] = paddings[4];
        paddings[7] = paddings[5];
        paddings[4] = paddings[2];
        paddings[5] = paddings[3];
        nvinfer1::Permutation perm;
        if (engine_->with_dynamic_shape()) {
          perm.order[0] = 0;
          perm.order[1] = 3;
          perm.order[2] = 1;
          perm.order[3] = 2;
        } else {
          perm.order[0] = 2;
          perm.order[1] = 0;
          perm.order[2] = 1;
        }

        auto* transpose_layer1 = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
        transpose_layer1->setFirstTranspose(perm);
        transpose_layer1_out = transpose_layer1->getOutput(0);
      }
    }

    int pad_size = static_cast<int>(paddings.size());

    nvinfer1::DimsHW pre_pad(paddings[pad_size - 4], paddings[pad_size - 2]);
    nvinfer1::DimsHW post_pad(paddings[pad_size - 3], paddings[pad_size - 1]);

    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_,
        Padding,
        transpose_layer1_out == nullptr ? *input : *transpose_layer1_out,
        pre_pad,
        post_pad);

    if (!flag_nchw) {
      nvinfer1::Permutation perm;

      if (engine_->with_dynamic_shape()) {
        perm.order[0] = 0;
        perm.order[1] = 2;
        perm.order[2] = 3;
        perm.order[3] = 1;
      } else {
        perm.order[0] = 1;
        perm.order[1] = 2;
        perm.order[2] = 0;
      }
      auto* transpose_layer2 =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(layer->getOutput(0)));
      transpose_layer2->setFirstTranspose(perm);
      PADDLE_ENFORCE_NOT_NULL(
          transpose_layer2,
          platform::errors::External(
              "add padding layer to tensorrt engine error"));
      auto output_name = op_desc.Output("Out")[0];
      RreplenishLayerAndOutput(
          transpose_layer2, "pad", {output_name}, test_mode);
    } else {
      PADDLE_ENFORCE_NOT_NULL(
          layer,
          platform::errors::External(
              "add padding layer to tensorrt engine error"));
      auto output_name = op_desc.Output("Out")[0];
      RreplenishLayerAndOutput(layer, "pad", {output_name}, test_mode);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(pad, PadOpConverter);
