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
#include "paddle/fluid/inference/tensorrt/plugin/elementwise_op_plugin.h"

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

class EqualOpConverter : public OpConverter {
 public:
  EqualOpConverter() {}
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
#if IS_TRT_VERSION_GE(8000)
    framework::OpDesc op_desc(op, nullptr);
    nvinfer1::ILayer* layer = nullptr;

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    auto* Y = engine_->GetITensor(op_desc.Input("Y").front());
    nvinfer1::Dims dims_x = X->getDimensions();
    nvinfer1::Dims dims_y = Y->getDimensions();

    int axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
    if (axis < 0) {
      axis = std::abs(dims_x.nbDims - dims_y.nbDims);
    }
    auto output_name = op_desc.Output("Out")[0];
    nvinfer1::IShuffleLayer* expand_layer = nullptr;
    if (dims_x.nbDims > dims_y.nbDims) {
      nvinfer1::Dims expand_shape;
      expand_shape.nbDims = dims_x.nbDims;
      for (int i = 0; i < expand_shape.nbDims; i++) {
        expand_shape.d[i] = 1;
      }
      for (int i = 0; i < dims_y.nbDims; i++) {
        expand_shape.d[i + axis] = dims_y.d[i];
      }
      expand_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *Y);
      expand_layer->setReshapeDimensions(expand_shape);
      Y = expand_layer->getOutput(0);
    } else if (dims_x.nbDims < dims_y.nbDims) {
      nvinfer1::Dims expand_shape;
      expand_shape.nbDims = dims_y.nbDims;
      for (int i = 0; i < expand_shape.nbDims; i++) {
        expand_shape.d[i] = 1;
      }
      for (int i = 0; i < dims_x.nbDims; i++) {
        expand_shape.d[i + axis] = dims_x.d[i];
      }
      expand_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *X);
      expand_layer->setReshapeDimensions(expand_shape);
      X = expand_layer->getOutput(0);
    }

    layer = TRT_ENGINE_ADD_LAYER(
        engine_, ElementWise, *X, *Y, nvinfer1::ElementWiseOperation::kEQUAL);
    RreplenishLayerAndOutput(layer, "equal", {output_name}, test_mode);
#else
    PADDLE_THROW(
        platform::errors::Fatal("ElementWise Equal Operation is only supported "
                                "on TRT 8 or higher version."));
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(equal, EqualOpConverter);
