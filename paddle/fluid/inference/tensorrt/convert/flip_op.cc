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

class FlipOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    vector<int> axis = op_desc.GetAttr("axis");

    // Need to flip the data input along the given axis using the Slice
    operator const auto input_shape = Shape(input);
    int intput_dims = input->getDimensions().nbDims;
    nvinfer1::ITensor* starts = Add1DConstantLayer(-1);
    nvinfer1::ITensor* ends = Add1DConstantLayer(static_cast<int64_t>(INT_MIN));
    nvinfer1::ITensor* axes = Add1DConstantLayer(axis);
    nvinfer1::ITensor* steps = Add1DConstantLayer(-1);

    if (axes->getDimensions().nbDims < intput_dims) {
      // axes specify a subset of the dimensions, or out of order.
      // Convert starts/ends/steps to complete in-order form.
      const nvinfer1::ITensor* subscripts{
          AxesToInterlaceSubscripts(axes, intput_dims)};
      starts = interlace(Similar(input_shape, 0), starts, subscripts);
      ends = interlace(input_shape, ends, subscripts);
      steps = interlace(Similar(input_shape, 1), steps, subscripts);
    }
    decodeOnnxStartsAndEnds(input_shape, steps, starts, ends);
    // TensorRT uses sizes of the output dimensions instead of ends.
    nvinfer1::ITensor* sizes =
        computeSliceSizes(ctx, starts, ends, steps, input_shape);

    nvinfer1::ISliceLayer* slice = addSlice(ctx, tensor, starts, sizes, steps);
    nvinfer1::ITensor* flippedTensor = *slice->getOutput(0);

    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    int dims = input->getDimensions().nbDims;
    nvinfer1::IShuffleLayer* layer = nullptr;
    if (!engine_->with_dynamic_shape()) {
      int dim_prod = 1;
      for (int i = 0; i < dims; i++) {
        int dim_i = input->getDimensions().d[i];
        PADDLE_ENFORCE_GT(
            dim_i,
            0,
            platform::errors::InvalidArgument(
                "flatten input dim should be > 0, but got %d.", dim_i));
        dim_prod *= dim_i;
      }
      nvinfer1::Dims flatten_dim;
      flatten_dim.nbDims = 1;
      flatten_dim.d[0] = dim_prod;
      layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      layer->setReshapeDimensions(flatten_dim);
    } else {
      auto* shape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shape, *input);
      nvinfer1::Dims start_dim, size_dim, stride_dim;
      start_dim.nbDims = 1;
      size_dim.nbDims = 1;
      stride_dim.nbDims = 1;
      start_dim.d[0] = 1;
      size_dim.d[0] = dims - 1;
      stride_dim.d[0] = 1;
      auto* slice_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                               Slice,
                                               *(shape_layer->getOutput(0)),
                                               start_dim,
                                               size_dim,
                                               stride_dim);
      uint32_t reduce_dim = 1;
      auto* reduce_prod_layer =
          TRT_ENGINE_ADD_LAYER(engine_,
                               Reduce,
                               *(slice_layer->getOutput(0)),
                               nvinfer1::ReduceOperation::kPROD,
                               reduce_dim,
                               true);
      int32_t* constant_weight_data = new int32_t[1];
      constant_weight_data[0] = -1;
      TensorRTEngine::Weight constant_weight{
          nvinfer1::DataType::kINT32,
          static_cast<void*>(constant_weight_data),
          1};
      nvinfer1::Dims constant_dims;
      constant_dims.nbDims = 1;
      constant_dims.d[0] = 1;
      auto* constant_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Constant, constant_dims, constant_weight.get());
      std::vector<nvinfer1::ITensor*> itensors;
      itensors.push_back(constant_layer->getOutput(0));
      itensors.push_back(reduce_prod_layer->getOutput(0));
      auto* concat_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Concatenation, itensors.data(), 2);
      concat_layer->setAxis(0);
      layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      layer->setInput(1, *(concat_layer->getOutput(0)));
    }
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "flatten", {output_name}, test_mode);
  }

  privete :

      nvinfer1::ITensor*
      Similar(nvinfer1::ITensor* exemplar) {}

  nvinfer1::ITensor* AxesToInterlaceSubscripts(const nvinfer1::ITensor* axes,
                                               int nbDims) {
    std::vector<int64_t> subscripts(nbDims);
    std::iota(subscripts.begin(), subscripts.end(), 0);
    for (int32_t i = 0; i < axes->getDimensions().nbDims; ++i) {
      subscripts[axes[i]] = nbDims + i;
    }
    return Add1DConstantLayer(subscripts);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(flip, FlipOpConverter);
