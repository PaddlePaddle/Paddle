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

class ExpandV2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a paddle expand_v2 op to trt expand layer.";
    framework::OpDesc op_desc(op, nullptr);
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto inputs = op_desc.Inputs();
    auto input_dims = input->getDimensions();
    auto output_name = op_desc.Output("Out")[0];
    auto rank = input_dims.nbDims;

    nvinfer1::ITensor* shape_tensor = nullptr;
    int32_t shape_rank = 0;
    if (inputs.find("Shape") != inputs.end() &&
        op_desc.Input("Shape").size() >= 1) {
      shape_tensor = engine_->GetITensor(op_desc.Input("Shape")[0]);
      shape_rank = shape_tensor->getDimensions().d[0];
    } else if (inputs.find("expand_shapes_tensor") != inputs.end() &&
               op_desc.Input("expand_shapes_tensor").size() >= 1) {
      int shape_size = op_desc.Input("expand_shapes_tensor").size();
      std::vector<nvinfer1::ITensor*> shape_tensors;
      for (int i = 0; i < shape_size; ++i) {
        shape_tensors.push_back(
            engine_->GetITensor(op_desc.Input("expand_shapes_tensor")[i]));
      }
      shape_tensor = Concat(shape_tensors);
      shape_rank = shape_size;
    } else {
      std::vector<int32_t> shape =
          PADDLE_GET_CONST(std::vector<int32_t>, op_desc.GetAttr("shape"));
      shape_tensor = Add1DConstantLayer(shape, output_name + "_shape_tensor_");
      shape_rank = shape.size();
    }

    nvinfer1::ITensor* input_shape_tensor;
    if (rank < shape_rank) {
      auto* one_rank_tensor =
          Add1DConstantLayer(std::vector<int32_t>(shape_rank - rank, 1),
                             output_name + "_one_rank_tensor_");
      auto in_shape_tensor = Shape(input);
      std::vector<nvinfer1::ITensor*> itensors;
      itensors.push_back(one_rank_tensor);
      itensors.push_back(in_shape_tensor);
      input_shape_tensor = Concat(itensors);
    } else {
      input_shape_tensor = Shape(input);
    }

    auto* shuffle = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    shuffle->setInput(1, *input_shape_tensor);

    std::vector<int32_t> start_vec(shape_rank, 0);
    nvinfer1::Dims start;
    start.nbDims = shape_rank;
    for (int32_t i = 0; i < shape_rank; ++i) {
      start.d[i] = start_vec[i];
    }
    nvinfer1::Dims size;
    size.nbDims = shape_rank;
    nvinfer1::Dims stride;
    stride.nbDims = shape_rank;

    auto starts_tensor =
        Add1DConstantLayer(start_vec, output_name + "_start_tensor_");
    auto one_tensor = Add1DConstantLayer(1, output_name + "_one_tensor_");

    auto sizes_tensor = Max(input_shape_tensor, shape_tensor);
    auto input_sub_tensor = Sub(input_shape_tensor, one_tensor);
    auto strides_tensor = Min(one_tensor, input_sub_tensor);

    auto layer = TRT_ENGINE_ADD_LAYER(
        engine_, Slice, *shuffle->getOutput(0), start, size, stride);
    layer->setInput(1, *starts_tensor);
    layer->setInput(2, *sizes_tensor);
    layer->setInput(3, *strides_tensor);

    RreplenishLayerAndOutput(layer, "expand_v2", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(expand_v2, ExpandV2OpConverter);
