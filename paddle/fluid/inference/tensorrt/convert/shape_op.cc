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

class ShapeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a shape op to tensorrt shape layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("Input")[0]);
    nvinfer1::ILayer* layer = TRT_ENGINE_ADD_LAYER(engine_, Shape, *input);
#if IS_TRT_VERSION_GE(10000)
    auto* cast_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Identity, *layer->getOutput(0));
    cast_layer->setOutputType(0, nvinfer1::DataType::kINT32);
    cast_layer->getOutput(0)->setType(nvinfer1::DataType::kINT32);
    layer = cast_layer;
#endif
    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "shape", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(shape, ShapeOpConverter);
