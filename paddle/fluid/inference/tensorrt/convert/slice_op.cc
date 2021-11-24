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
#include "paddle/fluid/inference/tensorrt/plugin/slice_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/special_slice_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class SliceOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    // This OP is implemented by trt dynamic shpae plugin.
    // Dynamic shape plugin requires TRT version greater than 6.0.
    VLOG(4) << "convert slice op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("Input")[0]);

    if (op_desc.HasAttr("out_threshold")) {
      float out_scale =
          BOOST_GET_CONST(float, op_desc.GetAttr("out_threshold"));
      engine_->SetTensorDynamicRange(input, out_scale);
    }

    std::vector<int> axes =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("axes"));
    std::vector<int> starts =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("starts"));
    std::vector<int> ends =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("ends"));

    auto input_dims = input->getDimensions();
    if (!engine_->with_dynamic_shape()) {
      // notice that input shape is [CHW] without batch axis when input has
      // static shape
      for (size_t i = input_dims.nbDims; i > 0; i--) {
        input_dims.d[i] = input_dims.d[i - 1];
      }
      input_dims.d[0] = 1;  // fake batchsize, not useful here
      for (size_t i = 0; i < axes.size(); i++) {
        if (starts[i] < 0) {
          starts[i] = std::max(starts[i] + input_dims.d[axes[i]], 0);
        }
        if (ends[i] < 0) {
          ends[i] = std::max(ends[i] + input_dims.d[axes[i]], 0);
        }
        ends[i] = std::min(ends[i], input_dims.d[axes[i]]);
        PADDLE_ENFORCE_GT(
            ends[i], starts[i],
            platform::errors::InvalidArgument(
                "Attr(ends) should be greater than attr(starts) in "
                "slice op. But received ends = %d, starts = %d.",
                ends[i], starts[i]));
      }
    }

    std::unordered_map<int, std::pair<int, int>> axes_starts_ends;
    for (size_t i = 0; i < axes.size(); i++) {
      axes_starts_ends[axes[i]] = std::make_pair(starts[i], ends[i]);
    }

    const auto in_dims = input->getDimensions();
    const int input_nbdims = in_dims.nbDims;
    nvinfer1::Dims start_dims;
    start_dims.nbDims = input_nbdims;
    nvinfer1::Dims size_dims;
    size_dims.nbDims = input_nbdims;
    nvinfer1::Dims stride_dims;
    stride_dims.nbDims = input_nbdims;

    for (int i = 0; i < input_nbdims; i++) {
      stride_dims.d[i] = 1;
    }

    const int offset = engine_->with_dynamic_shape() ? 0 : 1;
    for (int i = 0; i < input_nbdims; i++) {
      auto iter = axes_starts_ends.find(i + offset);
      if (iter != axes_starts_ends.end()) {
        start_dims.d[i] = iter->second.first;
        size_dims.d[i] = iter->second.second - iter->second.first;
      } else {
        start_dims.d[i] = 0;
        size_dims.d[i] = in_dims.d[i];
      }
    }

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, start_dims,
                                       size_dims, stride_dims);

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "slice", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(slice, SliceOpConverter);
