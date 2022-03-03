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
    auto dims = input->getDimensions();
    int start_axis = BOOST_GET_CONST(int, op_desc.GetAttr("start_axis"));
    int stop_axis = BOOST_GET_CONST(int, op_desc.GetAttr("stop_axis"));

    nvinfer1::IShuffleLayer* layer = nullptr;
    if (!engine_->with_dynamic_shape()) {
      if (start_axis < 0) {
        start_axis += dims.nbDims + 1;
      }
      if (start_axis == 0) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The flatten_contiguous_range'start_axis must != 0, when use "
            "static shape mode."));
      } else {
        start_axis--;
      }

      if (stop_axis < 0) {
        stop_axis += dims.nbDims + 1;
      }
      if (stop_axis == 0) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The flatten_contiguous_range'stop_axis must != 0, when use static "
            "shape mode."));
      } else {
        stop_axis--;
      }
    } else {
      if (start_axis < 0) {
        start_axis += dims.nbDims;
      }
      if (stop_axis < 0) {
        stop_axis += dims.nbDims;
      }
    }
    nvinfer1::Dims flatten_dim;
    flatten_dim.nbDims = dims.nbDims - (stop_axis - start_axis);
    for (int i = 0; i < flatten_dim.nbDims; i++) {
      flatten_dim.d[i] = 0;
    }
    flatten_dim.d[start_axis] = -1;
    layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    layer->setReshapeDimensions(flatten_dim);
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
