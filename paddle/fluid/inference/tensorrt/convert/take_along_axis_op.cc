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
 * TakeAlongAxis Op
 */
class TakeAlongAxisOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    // AddGatherV2 is supported by the trt version of 8.2.
#if IS_TRT_VERSION_GE(8200)
    VLOG(3) << "convert take_along_axis op to tensorrt take_along_axis layer";
    framework::OpDesc op_desc(op, nullptr);
    const auto input_tensor = engine_->GetITensor(op_desc.Input("Input")[0]);
    const auto index_tensor = engine_->GetITensor(op_desc.Input("Index")[0]);
    auto output_name = op_desc.Output("Result")[0];

    int axis = 0;
    if (op_desc.HasAttr("Axis")) {
      axis = PADDLE_GET_CONST(int, op_desc.GetAttr("Axis"));
    }
    auto input_dims = input_tensor->getDimensions();
    int NbDims = input_dims.nbDims;
    if (axis < 0) axis = axis + NbDims;

    auto layer = TRT_ENGINE_ADD_LAYER(engine_,
                                      GatherV2,
                                      *input_tensor,
                                      *index_tensor,
                                      nvinfer1::GatherMode::kELEMENT);
    layer->setGatherAxis(axis);

    ReplenishLayerAndOutput(layer, "take_along_axis", {output_name}, test_mode);
#endif
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(take_along_axis, TakeAlongAxisOpConverter);
