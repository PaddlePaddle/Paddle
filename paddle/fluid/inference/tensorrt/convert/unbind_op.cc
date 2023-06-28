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

namespace paddle {
namespace inference {
namespace tensorrt {


/*
 * Unbind Op
 */
class UnbindOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override{
    VLOG(3) << "convert a unbind op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);
    std::string input_x_name = op_desc.Input("X").front();
    auto* input_x_tensor = engine_->GetITensor(input_x_name);
    auto in_dims = input_x_tensor->getDimensions();
    auto rank = in_dims.nbDims;
    int axis = 0;
    std::cout << std::endl;
    if (op_desc.HasAttr("axis")) {
      axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
      if (axis < 0) {
        axis += rank;
      }
    }
    
    nvinfer1::Dims start;
    nvinfer1::Dims size;
    nvinfer1::Dims stride;
    std::vector<int32_t> newDims_vec;
    std::vector<int32_t> start_vec(rank, 0);
    std::vector<int32_t> size_vec(rank, 0);
    std::vector<int32_t> stride_vec(rank, 1);
    start.nbDims = rank;
    size.nbDims = rank;
    stride.nbDims = rank;
    for(int i = 0; i < rank; ++i){
        if(axis == i){
            size_vec[i] = 1;
        }else{
            size_vec[i] = in_dims.d[i];
            newDims_vec.push_back(in_dims.d[i]);
        }
    }
    auto sizes_tensor = Add1DConstantLayer(size_vec, input_x_name + "_size_tensor_");
    auto strides_tensor = Add1DConstantLayer(stride_vec, input_x_name + "_stride_tensor_");
    auto newDims_tensor = Add1DConstantLayer(newDims_vec, input_x_name + "_newDims_tensor_");
    for(auto& output_name : op_desc.Output("Out")){
        // 1 slice
        auto starts_tensor =Add1DConstantLayer(start_vec, output_name + "_start_tensor_");
        auto inputSliced = TRT_ENGINE_ADD_LAYER(
        engine_, Slice, *input_x_tensor, start, size, stride);
        inputSliced->setInput(1, *starts_tensor);
        inputSliced->setInput(2, *sizes_tensor);
        inputSliced->setInput(3, *strides_tensor);
        auto inputSliced_out = inputSliced->getOutput(0);
        // 2 reshape
        auto inputReshaped = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *inputSliced_out);
        inputReshaped->setInput(1, *newDims_tensor);
        RreplenishLayerAndOutput(inputReshaped, "unbind", {output_name}, test_mode);
        // 3 revise start dims
        start_vec[axis]++;
    }

  }
};



}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
REGISTER_TRT_OP_CONVERTER(unbind, UnbindOpConverter);