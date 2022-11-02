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

class ScatterNdOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    const auto input_tensor = engine_->GetITensor(op_desc.Input("X").front());
    const auto index_tensor =
        engine_->GetITensor(op_desc.Input("Index").front());
    const auto updates_tensor =
        engine_->GetITensor(op_desc.Input("Updates").front());
    auto output_name = op_desc.Output("Out").front();

    auto input_dims = input_tensor->getDimensions();
    auto index = index_tensor->getDimensions();
    auto layer = TRT_ENGINE_ADD_LAYER(engine_,
                                      Scatter,
                                      *input_tensor,
                                      *index_tensor,
                                      *updates_tensor,
                                      nvinfer1::ScatterMode::kND);
    layer->setAxis(0);

    RreplenishLayerAndOutput(layer, "scatter_nd_add", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(scatter_nd_add, ScatterNdOpConverter);
