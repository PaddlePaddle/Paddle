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



namespace paddle {
namespace inference {
namespace tensorrt {
// we use tensorrt ScatterElement to generate set value
// For example, if indices has dimensions [N,C,H,W] and axis is 2, then the
// updates happen as: for n in [0,n)
//     for c in [0,n)
//         for h in [0,n)
//             for w in [0,n)
//                 output[n,c,indices[n,c,h,w],w] = updates[n,c,h,w]]

class PutAlongAxisConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a put_along_axis op to tensorrt";
    framework::OpDesc op_desc(op, nullptr);

    auto* inputs = engine_->GetITensor(op_desc.Input("Input")[0]);
    auto* updates = engine_->GetITensor(op_desc.Input("Value")[0]);
    auto* index = engine_->GetITensor(op_desc.Input("Index")[0]);
    auto output_name = op_desc.Output("Result")[0];

    int axis = PADDLE_GET_CONST(int, op_desc.GetAttr("Axis"));

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                         Scatter,
                                         *inputs,
                                         *index,
                                         *updates,
                                         nvinfer1::ScatterMode::kELEMENT);

      layer->setAxis(axis);

      ReplenishLayerAndOutput(layer, "put_along_axis", {output_name}, test_mode);
     // std::cout << output_name << std::endl;
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(put_along_axis, PutAlongAxisConverter);
