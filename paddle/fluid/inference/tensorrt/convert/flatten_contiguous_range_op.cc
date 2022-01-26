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
 * flatten_contiguous_range trt converter
 */
class FlattenContiguousRangeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    int dims = input->getDimensions().nbDims;
    int start_axis = BOOST_GET_CONST(int, op_desc.GetAttr("start_axis"));
    int stop_axis = BOOST_GET_CONST(int, op_desc.GetAttr("stop_axis"));

    nvinfer1::IShuffleLayer* layer = nullptr;
    if (!engine_->with_dynamic_shape()) {
      if (start_axis < 0) start_axis += dims + 1;
      if (stop_axis < 0) stop_axis += dims + 1;
      int dim_prod = 1;
      nvinfer1::Dims flatten_dim;
      flatten_dim.nbDims = dims - (stop_axis - start_axis);
      for (int i = 0, j = 0; i < dims; ++i) {
        if (start_axis <= i + 1 && i + 1 <= stop_axis) {
          int dim_i = input->getDimensions().d[i];
          PADDLE_ENFORCE_GT(dim_i, 0, platform::errors::InvalidArgument(
                                          "flatten_contiguous_range input dim "
                                          "should be > 0, but got %d.",
                                          dim_i));
          dim_prod *= dim_i;
          if (i + 1 == stop_axis) {
            flatten_dim.d[j++] = dim_prod;
          }
        } else {
          flatten_dim.d[j++] = input->getDimensions().d[i];
        }
      }
      layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      layer->setReshapeDimensions(flatten_dim);
    } else {
      if (start_axis < 0) start_axis += dims;
      if (stop_axis < 0) stop_axis += dims;
      auto* shape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shape, *input);
      auto* shape_layer_itensor = shape_layer->getOutput(0);

      nvinfer1::Dims start_dim, size_dim, stride_dim;
      start_dim.nbDims = 1;
      size_dim.nbDims = 1;
      stride_dim.nbDims = 1;
      start_dim.d[0] = start_axis;
      size_dim.d[0] = stop_axis - start_axis + 1;
      stride_dim.d[0] = 1;
      auto* slice_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Slice, *shape_layer_itensor, start_dim,
                               size_dim, stride_dim);
      uint32_t reduce_dim = 1;
      auto* reduce_prod_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Reduce, *(slice_layer->getOutput(0)),
          nvinfer1::ReduceOperation::kPROD, reduce_dim, true);

      nvinfer1::ITensor* input_shape = nullptr;
      if (start_axis == 0 && stop_axis == dims - 1) {
        input_shape = reduce_prod_layer->getOutput(0);
      } else {
        std::vector<nvinfer1::ITensor*> itensors;
        if (start_axis > 0) {
          nvinfer1::Dims left_start_dim, left_size_dim, left_stride_dim;
          left_start_dim.nbDims = 1;
          left_size_dim.nbDims = 1;
          left_stride_dim.nbDims = 1;
          left_start_dim.d[0] = 0;
          left_size_dim.d[0] = start_axis;
          left_stride_dim.d[0] = 1;
          auto* slice_layer_left = TRT_ENGINE_ADD_LAYER(
              engine_, Slice, *shape_layer_itensor, left_start_dim,
              left_size_dim, left_stride_dim);
          itensors.push_back(slice_layer_left->getOutput(0));
        }
        itensors.push_back(reduce_prod_layer->getOutput(0));
        if (stop_axis < dims - 1) {
          nvinfer1::Dims right_start_dim, right_size_dim, right_stride_dim;
          right_start_dim.nbDims = 1;
          right_size_dim.nbDims = 1;
          right_stride_dim.nbDims = 1;
          right_start_dim.d[0] = stop_axis + 1;
          right_size_dim.d[0] = dims - stop_axis - 1;
          right_stride_dim.d[0] = 1;
          auto* slice_layer_right = TRT_ENGINE_ADD_LAYER(
              engine_, Slice, *shape_layer_itensor, right_start_dim,
              right_size_dim, right_stride_dim);
          itensors.push_back(slice_layer_right->getOutput(0));
        }
        auto* concat_layer = TRT_ENGINE_ADD_LAYER(
            engine_, Concatenation, itensors.data(), itensors.size());
        concat_layer->setAxis(0);
        input_shape = concat_layer->getOutput(0);
      }
      layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      layer->setInput(1, *input_shape);
    }
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "flatten_contiguous_range", {output_name},
                             test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(flatten_contiguous_range,
                          FlattenContiguousRangeOpConverter);
