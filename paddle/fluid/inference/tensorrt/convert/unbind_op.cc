/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
 * Unbind Op
 */
class UnbindOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a unbind op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);
    std::string input_x_name = op_desc.Input("X").front();
    auto* input_x_tensor = engine_->GetITensor(input_x_name);
    auto in_dims = input_x_tensor->getDimensions();
    auto in_shape_tensor = Shape(input_x_tensor);
    auto rank = in_dims.nbDims;
    int axis = 0;
    if (op_desc.HasAttr("axis")) {
      axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
      if (axis < 0) {
        axis += rank;
      }
    }

    std::vector<nvinfer1::ITensor*> in_shape_tensors;
    std::vector<nvinfer1::ITensor*> newDims_tensors;
    for (int32_t i = 0; i < rank; ++i) {
      in_shape_tensors.push_back(GetEleTensorOfShape(in_shape_tensor, i));
      if (i != axis) {
        newDims_tensors.push_back(GetEleTensorOfShape(in_shape_tensor, i));
      }
    }
    auto newDims_tensor = Concat(newDims_tensors);

    std::vector<nvinfer1::ITensor*> start_tensors;
    std::vector<nvinfer1::ITensor*> size_tensors = in_shape_tensors;

    nvinfer1::Dims stride;
    stride.nbDims = rank;
    for (int i = 0; i < rank; ++i) {
      if (axis == i) {
        size_tensors[i] = Add1DConstantLayer(1);
      }
      start_tensors.push_back(Add1DConstantLayer(0));
      stride.d[i] = 1;
    }
    int ii = 0;
    for (auto& output_name : op_desc.Output("Out")) {
      start_tensors[axis] = Add1DConstantLayer(ii++);
      // 1 slice
      auto inputSliced = TRT_ENGINE_ADD_LAYER(
          engine_, Slice, *input_x_tensor, stride, stride, stride);
      inputSliced->setInput(1, *Concat(start_tensors));
      inputSliced->setInput(2, *Concat(size_tensors));

      auto inputSliced_out = inputSliced->getOutput(0);
      // 2 reshape
      auto inputReshaped =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *inputSliced_out);
      inputReshaped->setInput(1, *newDims_tensor);
      ReplenishLayerAndOutput(
          inputReshaped, "unbind", {output_name}, test_mode);
    }
  }
};

}  // namespace paddle::inference::tensorrt
REGISTER_TRT_OP_CONVERTER(unbind, UnbindOpConverter);
