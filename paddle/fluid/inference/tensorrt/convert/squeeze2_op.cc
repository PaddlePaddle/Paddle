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

class Squeeze2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a squeeze2 op to tensorrt shuffle layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto input_dims = input->getDimensions();
    auto output_name = op_desc.Output("Out")[0];

    // Get Attrs
    std::vector<int> axes;
    if (op_desc.HasAttr("axes")) {
      axes = PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("axes"));
    }
    if (axes.empty()) {
      for (int i = 0; i < input_dims.nbDims; i++) {
        if (input_dims.d[i] == -1) {
          PADDLE_THROW(common::errors::InvalidArgument(
              "The necessary attributes of the squeeze2 operator axes is "
              "missing."));
        } else if (input_dims.d[i] == 1) {
          axes.push_back(i);
        }
      }
    }

    PADDLE_ENFORCE_GT(
        axes.size(),
        0,
        common::errors::InvalidArgument(
            "Attr(axes).size should be > 0 in squeeze2 op in TensorRT,"
            "but received axes.size() = %d.",
            axes.size()));

    std::vector<bool> should_squeeze(input_dims.nbDims, false);
    for (int& axis : axes) {
      axis += (axis < 0) ? input_dims.nbDims : 0;
      should_squeeze[axis] = true;
    }

    nvinfer1::Dims trt_out_dims;
    trt_out_dims.nbDims = 0;
    std::vector<int32_t> gather_indices;
    for (size_t i = 0; i < should_squeeze.size(); i++) {
      if (should_squeeze[i]) continue;
      gather_indices.push_back(i);
      // for static shape
      trt_out_dims.d[trt_out_dims.nbDims] = input_dims.d[i];
      trt_out_dims.nbDims++;
    }

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    auto* shape_tensor = Shape(input);
    auto* real_shape_tensor = Gather(shape_tensor, gather_indices);
    layer->setInput(1, *real_shape_tensor);
    ReplenishLayerAndOutput(layer, "squeeze2", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(squeeze2, Squeeze2OpConverter);
