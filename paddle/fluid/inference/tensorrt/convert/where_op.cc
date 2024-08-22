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

/*
 * Where Op
 */
class WhereOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a where op to tensorrt where layer";

    framework::OpDesc op_desc(op, nullptr);
    std::string input_x_name = op_desc.Input("X").front();
    std::string condition_name = op_desc.Input("Condition").front();
    std::string input_y_name = op_desc.Input("Y").front();
    std::string output_name = op_desc.Output("Out").front();

    const auto input_x_tensor = engine_->GetITensor(input_x_name);
    const auto condition_tensor = engine_->GetITensor(condition_name);
    const auto input_y_tensor = engine_->GetITensor(input_y_name);

    auto layer = TRT_ENGINE_ADD_LAYER(
        engine_, Select, *condition_tensor, *input_x_tensor, *input_y_tensor);

    ReplenishLayerAndOutput(layer, "where", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(where, WhereOpConverter);
