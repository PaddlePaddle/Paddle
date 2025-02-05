/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

class LookupTableOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    VLOG(3) << "convert lookup_table op to TensorRT IGatherLayer";

    auto ids_name = op_desc.Input("Ids").front();
    auto w_name = op_desc.Input("W").front();
    auto out_name = op_desc.Output("Out").front();

    auto* ids_tensor = engine_->GetITensor(ids_name);
    auto* w_tensor = engine_->GetITensor(w_name);

    std::vector<nvinfer1::ITensor*> after_shape_tensors;
    // lookup_table'Ids has an additional one-dimensional dimension (*,1), need
    // to reshape (*)
    for (int i = 0; i < ids_tensor->getDimensions().nbDims - 1; ++i) {
      after_shape_tensors.push_back(GetEleTensorOfShape(Shape(ids_tensor), i));
    }

    auto* reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *ids_tensor);
    reshape_layer->setInput(1, *Concat(after_shape_tensors));
    reshape_layer->setName(
        ("reshape Ids for lookup_table(Output: " + out_name + ")").c_str());

    auto* gather_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Gather, *w_tensor, *reshape_layer->getOutput(0), 0);
    ReplenishLayerAndOutput(gather_layer, "gather", {out_name}, test_mode);
  }
};

class LookupTableV2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    VLOG(3) << "convert lookup_table_v2 op to TensorRT IGatherLayer";

    auto ids_name = op_desc.Input("Ids").front();
    auto w_name = op_desc.Input("W").front();
    auto out_name = op_desc.Output("Out").front();

    auto* ids_tensor = engine_->GetITensor(ids_name);
    auto* w_tensor = engine_->GetITensor(w_name);

    auto* gather_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Gather, *w_tensor, *ids_tensor, 0);
    ReplenishLayerAndOutput(gather_layer, "gather", {out_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(lookup_table, LookupTableOpConverter);
REGISTER_TRT_OP_CONVERTER(lookup_table_v2, LookupTableV2OpConverter);
