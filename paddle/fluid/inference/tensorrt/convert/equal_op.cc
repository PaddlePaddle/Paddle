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

namespace paddle::inference::tensorrt {

class EqualOpConverter : public OpConverter {
 public:
  EqualOpConverter() = default;
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert equal op to tensorrt layer";
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
    ReplenishLayerAndOutput(layer, "equal", {output_name}, test_mode);
  }
};

class NotEqualOpConverter : public OpConverter {
 public:
  NotEqualOpConverter() = default;
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert not_equal op to tensorrt layer";
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

    layer = TRT_ENGINE_ADD_LAYER(
        engine_, Unary, *layer->getOutput(0), nvinfer1::UnaryOperation::kNOT);

    ReplenishLayerAndOutput(layer, "not_equal", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(equal, EqualOpConverter);
REGISTER_TRT_OP_CONVERTER(not_equal, NotEqualOpConverter);
