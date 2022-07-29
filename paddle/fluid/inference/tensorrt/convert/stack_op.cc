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
#include "paddle/fluid/inference/tensorrt/plugin/stack_op_plugin.h"

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
 * Stack converter from fluid to tensorRT.
 */
class StackOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert fluid stack op to tensorrt stack layer";

    framework::OpDesc op_desc(op, nullptr);
    auto input = op_desc.Input("X");
    int input_num = input.size();
    std::vector<nvinfer1::ITensor*> inputs;

    for (int i = 0; i < input_num; ++i) {
      inputs.push_back(engine_->GetITensor(input[i]));
      if (op_desc.HasAttr("out_threshold")) {
        float out_scale =
            PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
        engine_->SetTensorDynamicRange(inputs[i], out_scale);
      }
    }

    int axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
    if (axis < 0) {
      axis = axis + inputs[0]->getDimensions().nbDims + 1;
    }

    auto* concat_layer = TRT_ENGINE_ADD_LAYER(engine_, Concatenation, inputs.data(), inputs.size());
    // note now, axis is relative to output but not input, so we
    concat_layer->setAxis(axis == 0 ? 0 : axis - 1);


    auto* shape_tensor = Shape(inputs[0]);
    std::vector<int32_t> gather_index;
    std::vector<nvinfer1::ITensor*> shape_concat = {shape_tensor, Add1DConstantLayer(input_num)};
    for (int i = 0; i < inputs[0]->getDimensions().nbDims + 1; i++) {
       if(i < axis) {
         gather_index.push_back(i);
       } else if(i > axis) {
         gather_index.push_back(i - 1);
       } else {
         gather_index.push_back(inputs[0]->getDimensions().nbDims);
       }
    }

    auto* reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *concat_layer->getOutput(0));
    reshape_layer->setInput(1, *Gather(Concat(shape_concat), gather_index));

    auto output_name = op_desc.Output("Y").front();
    RreplenishLayerAndOutput(reshape_layer, "reshape_after_stack", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(stack, StackOpConverter);
