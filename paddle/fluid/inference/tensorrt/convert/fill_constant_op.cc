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

namespace paddle {
namespace inference {
namespace tensorrt {

class FillConstantOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4)
        << "convert a fluid fill_constant op to tensorrt fill_constant layer";

    framework::OpDesc op_desc(op, nullptr);
    int dtype = BOOST_GET_CONST(int, op_desc.GetAttr("dtype"));
    std::string str_value =
        BOOST_GET_CONST(std::string, op_desc.GetAttr("str_value"));
    std::vector<int64_t> shape =
        BOOST_GET_CONST(std::vector<int64_t>, op_desc.GetAttr("shape"));
    std::unique_ptr<framework::Tensor> paddle_out_tensor(
        new framework::Tensor());
    paddle_out_tensor->Resize(phi::make_ddim(shape));

    nvinfer1::DataType trt_dtype = nvinfer1::DataType::kINT32;
    void* trt_data = nullptr;
    size_t trt_num;
    if (dtype == 2) {  // int
      auto* tmp_ptr =
          paddle_out_tensor->mutable_data<int>(platform::CPUPlace());
      for (int64_t i = 0; i < paddle_out_tensor->numel(); i++)
        tmp_ptr[i] = std::stoi(str_value);
      trt_dtype = nvinfer1::DataType::kINT32;
      trt_data = static_cast<void*>(tmp_ptr);
    } else if (dtype == 5) {  // float
      auto* tmp_ptr =
          paddle_out_tensor->mutable_data<float>(platform::CPUPlace());
      for (int64_t i = 0; i < paddle_out_tensor->numel(); i++)
        tmp_ptr[i] = std::stof(str_value);
      trt_dtype = nvinfer1::DataType::kFLOAT;
      trt_data = static_cast<void*>(tmp_ptr);
    }
    trt_num = static_cast<size_t>(paddle_out_tensor->numel());
    engine_->SetWeights("fill_constant_value", std::move(paddle_out_tensor));
    TensorRTEngine::Weight weight{trt_dtype, trt_data, trt_num};

    nvinfer1::Dims trt_in_shape;
    trt_in_shape.nbDims = shape.size();
    for (size_t i = 0; i < shape.size(); i++) trt_in_shape.d[i] = shape[i];

    nvinfer1::ILayer* layer =
        TRT_ENGINE_ADD_LAYER(engine_, Constant, trt_in_shape, weight.get());
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "fill_constant", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fill_constant, FillConstantOpConverter);
