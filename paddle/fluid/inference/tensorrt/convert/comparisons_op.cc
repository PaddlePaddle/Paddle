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

class ComparisonsOpConverter : public OpConverter {
 public:
  ComparisonsOpConverter() {}
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    auto op_pair = ops.find(op_type_);
    PADDLE_ENFORCE_NE(op_pair, ops.end(),
                      platform::errors::InvalidArgument(
                          "Elementwise op's type(%s) is not supported. Please "
                          "check if the op_type is correct.",
                          op_type_));

    // Here the two nullptr looks strange, that's because the
    // framework::OpDesc's constructor is strange.
    framework::OpDesc op_desc(op, nullptr);
    nvinfer1::ILayer* layer = nullptr;

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    auto* Y = engine_->GetITensor(op_desc.Input("Y").front());
    nvinfer1::Dims dims_x = X->getDimensions();
    nvinfer1::Dims dims_y = Y->getDimensions();

    int axis = BOOST_GET_CONST(int, op_desc.GetAttr("axis"));
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

    layer = TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *X, *Y, op_pair->second);
    RreplenishLayerAndOutput(layer, "elementwise", {output_name}, test_mode);
  }

 protected:
  static const std::unordered_map<std::string, nvinfer1::ElementWiseOperation>
      ops;
  std::string op_type_;
};

#if IS_TRT_VERSION_GE(7000)
const std::unordered_map<std::string, nvinfer1::ElementWiseOperation>
    ComparisonsOpConverter::ops = {
        {"greater", nvinfer1::ElementWiseOperation::kGREATER},
        {"less", nvinfer1::ElementWiseOperation::kLESS},
        {"equal", nvinfer1::ElementWiseOperation::kEQUAL},
};
#else
PADDLE_THROW(
    platform::errors::Fatal("ElementWise Compare Operation is only supported "
                            "on TRT 7 or higher version."));
#endif

class GreaterOpConverter : public ComparisonsOpConverter {
 public:
  GreaterOpConverter() { op_type_ = "greater"; }
};

class LessOpConverter : public ComparisonsOpConverter {
 public:
  LessOpConverter() { op_type_ = "less"; }
};

class EqualOpConverter : public ComparisonsOpConverter {
 public:
  EqualOpConverter() { op_type_ = "equal"; }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(greater, GreaterOpConverter);
REGISTER_TRT_OP_CONVERTER(less, LessOpConverter);
REGISTER_TRT_OP_CONVERTER(equal, EqualOpConverter);
