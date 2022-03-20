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
 * LookupTable Op
 */
class LookupTableOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid lookup_table op to tensorrt gather layer";

    framework::OpDesc op_desc(op, nullptr);
    std::string ids_name = op_desc.Input("Ids").front();
    std::string w_name = op_desc.Input("W").front();
    std::string output_name = op_desc.Output("Out").front();
    const auto ids_tensor = engine_->GetITensor(ids_name);

    // The rank of ids is 2 in implicit batch mode, and the last dim is 1,
    // reshape the ids to 1-dimensional tensor.
    auto di = ids_tensor->getDimensions();
    for (int i = 0; i < di.nbDims; i++) {
      std::cout << "ids shape: " << di.d[i] << std::endl;
    }

    nvinfer1::Dims shape{};
    shape.nbDims = 1;
    shape.d[0] = -1;
    auto reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *ids_tensor);
    reshape_layer->setReshapeDimensions(shape);
    reshape_layer->setName(
        ("LookupTable: Shuffle: (Output: " + output_name + ")").c_str());

    auto dii = reshape_layer->getOutput(0)->getDimensions();
    for (int i = 0; i < dii.nbDims; i++) {
      std::cout << "reshape ids shape: " << dii.d[i] << std::endl;
    }

    auto* weight_v = scope.FindVar(w_name);
    PADDLE_ENFORCE_NOT_NULL(
        weight_v, platform::errors::NotFound(
                      "Can not find %s presistale var in scope.", w_name));

    auto* weight_t = weight_v->GetMutable<framework::LoDTensor>();
    float* weight_data = engine_->GetWeightCPUData(w_name, weight_t, false);

    TensorRTEngine::Weight weights{nvinfer1::DataType::kFLOAT,
                                   static_cast<void*>(weight_data),
                                   static_cast<size_t>(weight_t->numel())};

    auto weight_dims = weight_t->dims();
    nvinfer1::Dims dims;
    dims.nbDims = weight_dims.size();
    for (int i = 0; i < weight_dims.size(); i++) {
      dims.d[i] = weight_dims.at(i);
    }

    // convert weight to ITensor, ITensor is interface
    auto* constant_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Constant, dims, weights.get());
    constant_layer->setName(
        ("LookupTable: Constant: (Output: " + output_name + ")").c_str());

    int axis = 0;
    int nb_elt_ims = 0;
    auto* layer =
        TRT_ENGINE_ADD_LAYER(engine_, Gather, *constant_layer->getOutput(0),
                             *reshape_layer->getOutput(0), axis);
    layer->setNbElementWiseDims(nb_elt_ims);
    layer->setName(
        ("LookupTable: Gather: (Output: " + output_name + ")").c_str());
    std::cout << "axis: " << layer->getGatherAxis() << std::endl;

#if IS_TRT_VERSION_GE(8200)
    layer->setMode(nvinfer1::GatherMode::kDEFAULT);
#endif

    RreplenishLayerAndOutput(layer, "lookup_table", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(lookup_table, LookupTableOpConverter);
