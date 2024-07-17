/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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
 * IndexPut Op
 */
class IndexPutOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a index_put op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    std::string input_name = op_desc.Input("x").front();
    std::string indices_name = op_desc.Input("indices").front();
    std::string value_name = op_desc.Input("value").front();
    std::string output_name = op_desc.Output("out").front();

    auto* input_tensor = engine_->GetITensor(input_name);
    auto* indices_tensor = engine_->GetITensor(indices_name);
    auto* value_tensor = engine_->GetITensor(value_name);
    auto* input_shape_tensor = Shape(input_tensor);
    nvinfer1::Dims input_dims = input_tensor->getDimensions();
    auto rank = input_dims.nbDims;

    nvinfer1::Dims indices_dims = indices_tensor->getDimensions();
    nvinfer1::Dims value_dims = value_tensor->getDimensions();

    // indices reshape
    std::vector<nvinfer1::ITensor*> indices_shape_vec;
    for (int i = 0; i < rank; ++i) {
      int indices_one = i < indices_dims.nbDims ? indices_dims.d[i] : 1;
      indices_shape_vec.push_back(Add1DConstantLayer(indices_one));
    }
    nvinfer1::ITensor* indices_shape_tensor_new = Concat(indices_shape_vec);
    nvinfer1::ITensor* indices_tensor_temp =
        Reshape(indices_tensor, indices_shape_tensor_new);
    // value reshape
    std::vector<nvinfer1::ITensor*> value_shape_vec;
    for (int i = 0; i < rank; ++i) {
      int shape_one = i < value_dims.nbDims ? value_dims.d[i] : 1;
      value_shape_vec.push_back(Add1DConstantLayer(shape_one));
    }
    nvinfer1::ITensor* value_shape_tensor_new = Concat(value_shape_vec);
    nvinfer1::ITensor* value_tensor_temp =
        Reshape(value_tensor, value_shape_tensor_new);

    // slice
    nvinfer1::Dims stride;
    stride.nbDims = input_dims.nbDims;
    for (int i = 0; i < stride.nbDims; ++i) {
      stride.d[i] = 1;
    }
    std::vector<nvinfer1::ITensor*> start_tensor_vec;
    std::vector<nvinfer1::ITensor*> stride_tensor_vec;
    for (int i = 0; i < rank; ++i) {
      start_tensor_vec.push_back(Add1DConstantLayer(0));
      stride_tensor_vec.push_back(Add1DConstantLayer(1));
    }
    auto* start_tensor = Concat(start_tensor_vec);
    auto* stride_tensor = Concat(stride_tensor_vec);

    auto indices_slice_layer = TRT_ENGINE_ADD_LAYER(
        engine_,
        Slice,
        *Cast(indices_tensor_temp, nvinfer1::DataType::kFLOAT),
        stride,
        stride,
        stride);
    indices_slice_layer->setInput(1, *start_tensor);
    indices_slice_layer->setInput(2, *input_shape_tensor);
    indices_slice_layer->setInput(3, *stride_tensor);
    indices_slice_layer->setMode(nvinfer1::SliceMode::kCLAMP);
    indices_slice_layer->setName("indices_slice_layer");
    auto* indices_tensor_new =
        Cast(indices_slice_layer->getOutput(0), nvinfer1::DataType::kBOOL);

    // value
    auto value_slice_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Slice, *value_tensor_temp, stride, stride, stride);
    value_slice_layer->setName("value_slice_layer");
    value_slice_layer->setInput(1, *start_tensor);
    value_slice_layer->setInput(2, *input_shape_tensor);
    value_slice_layer->setInput(3, *stride_tensor);
    value_slice_layer->setMode(nvinfer1::SliceMode::kCLAMP);
    auto* value_tensor_new = value_slice_layer->getOutput(0);

    auto layer = TRT_ENGINE_ADD_LAYER(
        engine_, Select, *indices_tensor_new, *value_tensor_new, *input_tensor);

    ReplenishLayerAndOutput(layer, "index_put", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(index_put, IndexPutOpConverter);
