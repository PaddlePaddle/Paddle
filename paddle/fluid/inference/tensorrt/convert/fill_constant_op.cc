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

class FillConstantOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fill_constant op to tensorrt fill_constant layer";

    framework::OpDesc op_desc(op, nullptr);
    int dtype = PADDLE_GET_CONST(int, op_desc.GetAttr("dtype"));
    std::string str_value =
        PADDLE_GET_CONST(std::string, op_desc.GetAttr("str_value"));
    if (str_value.empty()) {
      float value = PADDLE_GET_CONST(float, op_desc.GetAttr("value"));
      str_value = std::to_string(value);
    }
    nvinfer1::ILayer* layer = nullptr;
    if ((op_desc.HasInput("ShapeTensor") &&
         op_desc.Input("ShapeTensor").size() == 1) ||
        (op_desc.HasInput("ShapeTensorList") &&
         op_desc.Input("ShapeTensorList").size() >= 1)) {
      nvinfer1::ITensor* shapes_tensor;
      int tensor_rank = 0;
      if (op_desc.HasInput("ShapeTensor") &&
          op_desc.Input("ShapeTensor").size() == 1) {
        shapes_tensor = engine_->GetITensor(op_desc.Input("ShapeTensor")[0]);
        auto shape_nbDims = shapes_tensor->getDimensions().nbDims;
        PADDLE_ENFORCE_EQ(shape_nbDims,
                          1,
                          common::errors::InvalidArgument(
                              "ShapeTensor nbDims must be 1, but received %d.",
                              shape_nbDims));
        tensor_rank = shapes_tensor->getDimensions().d[0];
      } else {
        int32_t shape_size = op_desc.Input("ShapeTensorList").size();
        std::vector<nvinfer1::ITensor*> shape_tensor;
        nvinfer1::Dims dims{1, {1}};
        for (int32_t i = 0; i < shape_size; ++i) {
          if (engine_->GetITensor(op_desc.Input("ShapeTensorList")[i])
                  ->getDimensions()
                  .nbDims == 0) {
            shape_tensor.push_back(Reshape(
                engine_->GetITensor(op_desc.Input("ShapeTensorList")[i]),
                dims));
          } else {
            shape_tensor.push_back(
                engine_->GetITensor(op_desc.Input("ShapeTensorList")[i]));
          }
        }
        tensor_rank = shape_size;
        shapes_tensor = Concat(shape_tensor, 0);
      }

      layer = TRT_ENGINE_ADD_LAYER(
          engine_, Fill, nvinfer1::Dims{}, nvinfer1::FillOperation::kLINSPACE);
      layer->setInput(0, *shapes_tensor);

      if (dtype == 2 || dtype == 3) {
        int value = std::stoi(str_value);
        std::vector<int> beta_vec(tensor_rank);
        std::vector<int> value_vec(1, value);
        layer->setInput(1, *Add1DConstantLayer(value_vec, "value_vec", true));
        layer->setInput(2, *Add1DConstantLayer(beta_vec, "beta_vec", false));
      } else if (dtype == 5) {  // float
        float value = std::stof(str_value);
        std::vector<float> beta_vec(tensor_rank);
        std::vector<float> value_vec(1, value);
        layer->setInput(1, *Add1DConstantLayer(value_vec, "value_vec", true));
        layer->setInput(2, *Add1DConstantLayer(beta_vec, "beta_vec", false));
      }
    } else {
      std::vector<int64_t> shape =
          PADDLE_GET_CONST(std::vector<int64_t>, op_desc.GetAttr("shape"));

      std::unique_ptr<phi::DenseTensor> out_tensor(new phi::DenseTensor());
      out_tensor->Resize(common::make_ddim(shape));
      nvinfer1::DataType trt_dtype = nvinfer1::DataType::kFLOAT;
      void* trt_data = nullptr;
      size_t trt_num;
      if (dtype == 2 || dtype == 3) {  // int,int64
        auto* tmp_ptr = out_tensor->mutable_data<int>(phi::CPUPlace());
        for (int64_t i = 0; i < out_tensor->numel(); i++)
          tmp_ptr[i] = std::stoi(str_value);
        trt_dtype = nvinfer1::DataType::kINT32;
        trt_data = static_cast<void*>(tmp_ptr);
      } else if (dtype == 5) {  // float
        auto* tmp_ptr = out_tensor->mutable_data<float>(phi::CPUPlace());
        for (int64_t i = 0; i < out_tensor->numel(); i++)
          tmp_ptr[i] = std::stof(str_value);
        trt_data = static_cast<void*>(tmp_ptr);
      }

      trt_num = static_cast<size_t>(out_tensor->numel());
      engine_->SetWeights("fill_constant_value", std::move(out_tensor));
      TensorRTEngine::Weight weight{trt_dtype, trt_data, trt_num};

      nvinfer1::Dims trt_in_shape;
      trt_in_shape.nbDims = shape.size();
      for (size_t i = 0; i < shape.size(); i++) trt_in_shape.d[i] = shape[i];
      layer =
          TRT_ENGINE_ADD_LAYER(engine_, Constant, trt_in_shape, weight.get());
    }
    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "fill_constant", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(fill_constant, FillConstantOpConverter);
