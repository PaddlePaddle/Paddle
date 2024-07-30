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
 * Argsort Op
 */
class ArgsortOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a argsort op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    std::string input_name = op_desc.Input("X").front();
    std::string output_name = op_desc.Output("Out").front();
    std::string indices_name = op_desc.Output("Indices").front();
    int axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
    bool descending = PADDLE_GET_CONST(bool, op_desc.GetAttr("descending"));
    auto* input_tensor = engine_->GetITensor(input_name);
    nvinfer1::Dims input_tensor_dims = input_tensor->getDimensions();
    nvinfer1::TopKOperation operation = nvinfer1::TopKOperation::kMIN;
    if (descending) {
      operation = nvinfer1::TopKOperation::kMAX;
    }
    if (axis < 0) {
      axis += input_tensor_dims.nbDims;
    }
    nvinfer1::DataType in_type = input_tensor->getType();
    bool need_cast = in_type != nvinfer1::DataType::kFLOAT ? true : false;
    int x_rank = input_tensor->getDimensions().nbDims;
    if (x_rank == 1) {
      nvinfer1::Dims unsqueeze_shape;
      unsqueeze_shape.nbDims = 2;
      unsqueeze_shape.d[0] = 1;
      unsqueeze_shape.d[1] = -1;
      input_tensor = Reshape(input_tensor, unsqueeze_shape);
      axis = 1;
    }

    if (need_cast) {
      auto* cast_layer1 =
          TRT_ENGINE_ADD_LAYER(engine_, Identity, *input_tensor);
      cast_layer1->setOutputType(0, nvinfer1::DataType::kFLOAT);
      cast_layer1->getOutput(0)->setType(nvinfer1::DataType::kFLOAT);
      input_tensor = cast_layer1->getOutput(0);
    }

    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, TopK, *input_tensor, operation, 0, 1 << axis);
    auto* shape = Shape(input_tensor);
    auto* k_tensor = GetEleTensorOfShape(shape, axis, true);
    layer->setInput(1, *k_tensor);
    auto* Out = layer->getOutput(0);
    auto* Indices = layer->getOutput(1);
    if (need_cast) {
      auto* tmp_output = layer->getOutput(0);

      auto* cast_layer2 = TRT_ENGINE_ADD_LAYER(engine_, Identity, *tmp_output);
      cast_layer2->setOutputType(0, in_type);
      cast_layer2->getOutput(0)->setType(in_type);

      Out = cast_layer2->getOutput(0);
      Indices = layer->getOutput(1);
    }
    if (x_rank == 1) {
      nvinfer1::Dims squeeze_shape;
      squeeze_shape.nbDims = 1;
      squeeze_shape.d[0] = -1;
      Out = Reshape(Out, squeeze_shape);
      Indices = Reshape(Indices, squeeze_shape);
    }

    std::string layer_name = "argsort (Output: ";
    Out->setName(output_name.c_str());
    engine_->SetITensor(output_name, Out);
    layer_name += output_name + ", ";

    Indices->setName(indices_name.c_str());
    engine_->SetITensor(indices_name, Indices);
    layer_name += indices_name;

    layer->setName((layer_name + ")").c_str());
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(argsort, ArgsortOpConverter);
