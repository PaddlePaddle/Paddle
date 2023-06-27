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
// #if IS_TRT_VERSION_GE(7220)
    VLOG(3) << "convert a unbind op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);
    std::string input_x_name = op_desc.Input("X");
    // std::string output_name = op_desc.Output("Out").front();
    // std::string axis = op_desc.Input("axis");
    // int axis = ctx->Attrs().Get<int>("axis");
    auto* input_x_tensor = engine_->GetITensor(input_x_name);
    auto in_dims = input_x_tensor->getDimensions();
    auto rank = in_dims.nbDims;
    int axis =0;
    
    if (op_desc.HasAttr("axis")) {
      axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
      if (axis < 0) {
        axis += rank;
      }
    }
    // axis = [0, rank-1]
    // 输出tensor dim
    nvinfer1::Dims start;
    nvinfer1::Dims size;
    nvinfer1::Dims stride;
    std::vector<int32_t> newDims_vec;
    newDims_vec.reserve(rank - 1);
    start.nbDims = rank;
    size.nDims = rank;
    stride.nDims = rank;
    for(int i = 0; i < rank; ++i){
        start.d[i] = 0;
        stride.d[i] = 1;
        if(axis == i){
            size.d[i] = 1;
        }else{
            size.d[i] = in_dims.d[i];
            newDims_vec[i].push_back(in_dims.d[i]);
        }
    }
    // reshape tensor
    auto newDims = Add1DConstantLayer(start_vec, input_x_name + "_newDims_tensor_");
    for(int i = 0; i < in_dims.d[axis]; ++i){
        auto& output_name = op_desc.Output("Out")[i];

        // 1 slice
        auto inputSliced = TRT_ENGINE_ADD_LAYER(
        engine_, Slice, *input_x_tensor, start, size, stride);
        
        // 2 reshape
        auto inputSliced_output = Reshape(inputSliced, newDims, ("unbind: reshape: (Output(" + output_name + ")").c_str());
        
        RreplenishLayerAndOutput(inputSliced_output, "unbind", {output_name}, test_mode);
        // 3 revise start dims
        ++start.d[axis];
    }

    // std::vector<nvinfer1::ITensor*> itensors;
    // std::vecotr<std::string> output_names;
    // for (auto& output_name : op_desc.Output("X")) {
    //   output_names.push_back(output_name);
    // }
    
// #else
//     VLOG(3) << "Unbind is not supported when TensorRT < 7.2.2";
// #endif
  }
};



}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
REGISTER_TRT_OP_CONVERTER(unbind, UnbindOpConverter);