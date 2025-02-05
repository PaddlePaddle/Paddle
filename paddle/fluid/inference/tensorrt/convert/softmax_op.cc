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

#include <algorithm>

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle::inference::tensorrt {

/*
 * SoftMaxOp, ISoftMaxLayer in TRT. This Layer doesn't has weights.
 */
class SoftMaxOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a softmax op to tensorrt softmax layer without bias";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);
    nvinfer1::Dims input_shape = input1->getDimensions();
    int input_dims = input_shape.nbDims;
    int axis = op_desc.HasAttr("axis")
                   ? PADDLE_GET_CONST(int, op_desc.GetAttr("axis"))
                   : -1;

    // support 0 or 1 dims input
    bool is_0_dims = input_dims == 0;
    bool is_1_dims = input_dims == 1;
    if (is_0_dims || is_1_dims) {
      auto reshaped_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input1);
      nvinfer1::Dims reshaped_dims;
      reshaped_dims.nbDims = 2;
      reshaped_dims.d[0] = 1;
      reshaped_dims.d[1] = is_0_dims ? 1 : input_shape.d[0];
      reshaped_layer->setReshapeDimensions(reshaped_dims);
      input1 = reshaped_layer->getOutput(0);
      input_shape = input1->getDimensions();
      input_dims = input_shape.nbDims;
      axis = -1;
    }

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, SoftMax, *input1);
    uint32_t axes = std::max(0, input_dims - 3);
    // TODO(cryoco): Poor workaround. Fix padded dims problem when TRT layers
    // support Nd.
    // Tips: Dynamic shape already fixes.
    int padded_dims = 0;
    int explicit_batch = 1;
    for (int i = input_dims - 1; i > explicit_batch; i--) {
      if (input_shape.d[i] == 1) {
        padded_dims += 1;
      } else {
        break;
      }
    }
    if (axis < 0) {
      axes = input_dims + axis;
    } else {
      axes = axis;
    }
    layer->setAxes(1 << axes);

    auto output_name = op_desc.Output("Out")[0];

    // support 0 or 1 dims input
    if (is_0_dims || is_1_dims) {
      auto reshaped_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *layer->getOutput(0));
      reshaped_layer->setReshapeDimensions(
          engine_->GetITensor(op_desc.Input("X")[0])->getDimensions());
      ReplenishLayerAndOutput(
          reshaped_layer, "reshape_softmax_reshape", {output_name}, test_mode);
    } else {
      ReplenishLayerAndOutput(layer, "softmax", {output_name}, test_mode);
    }
  }
};

}  // namespace paddle::inference::tensorrt

USE_OP_ITSELF(softmax);
REGISTER_TRT_OP_CONVERTER(softmax, SoftMaxOpConverter);
