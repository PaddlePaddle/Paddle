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
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fill_constant op to tensorrt fill_constant layer";

    framework::OpDesc op_desc(op, nullptr);
    if (op_desc.HasInput("ShapeTensorList") &&
        op_desc.Input("ShapeTensorList").size()) {
      int dtype = PADDLE_GET_CONST(int, op_desc.GetAttr("dtype"));
      std::string str_value =
          PADDLE_GET_CONST(std::string, op_desc.GetAttr("str_value"));
      auto output_name = op_desc.Output("Out")[0];
      std::vector<std::string> input_names = op_desc.Input("ShapeTensorList");
      std::vector<nvinfer1::ITensor*> output_shape_tensors;

      if (str_value == "") {
        float value = PADDLE_GET_CONST(float, op_desc.GetAttr("value"));
        str_value = std::to_string(value);
      }
      if (str_value == "inf" || str_value == "Infinity") {
        str_value = "50000";
        VLOG(3) << "The fill_constant's attr str_value or value is inf, trt "
                   "will change to 50000.";
      } else if (str_value == "-inf" || str_value == "-Infinity") {
        str_value = "-50000";
        VLOG(3) << "The fill_constant's attr str_value or value is -inf, trt "
                   "will change to -50000.";
      }

      float value = std::stof(str_value);
      for (auto input_name : input_names) {
        auto input_tensor = engine_->GetITensor(input_name);
        output_shape_tensors.push_back(input_tensor);
      }
      auto output_shape_tensor = Concat(output_shape_tensors);
      nvinfer1::ITensor* value_tensor;
      nvinfer1::ITensor* beta_tensor;

      if (dtype == 2 || dtype == 3) {
        value_tensor = Add1DConstantLayer(
            std::vector<int32_t>(1, static_cast<int32_t>(value)),
            output_name + "_value_tensor_",
            true);
        beta_tensor =
            Add1DConstantLayer(std::vector<int32_t>(input_names.size(), 0),
                               output_name + "_beta_tensor_");
      } else {
        value_tensor = Add1DConstantLayer(
            std::vector<float>(1, value), output_name + "_value_tensor_", true);
        beta_tensor =
            Add1DConstantLayer(std::vector<float>(input_names.size(), 0),
                               output_name + "_beta_tensor_");
      }
      auto layer = TRT_ENGINE_ADD_LAYER(
          engine_, Fill, nvinfer1::Dims{}, nvinfer1::FillOperation::kLINSPACE);
      layer->setAlpha(value);
      layer->setBeta(0.0);
      layer->setInput(0, *output_shape_tensor);
      layer->setInput(1, *value_tensor);
      layer->setInput(2, *beta_tensor);
      RreplenishLayerAndOutput(
          layer, "fill_constant", {output_name}, test_mode);

    } else {
      int dtype = PADDLE_GET_CONST(int, op_desc.GetAttr("dtype"));
      std::string str_value =
          PADDLE_GET_CONST(std::string, op_desc.GetAttr("str_value"));
      std::vector<int64_t> shape =
          PADDLE_GET_CONST(std::vector<int64_t>, op_desc.GetAttr("shape"));
      if (str_value == "") {
        float value = PADDLE_GET_CONST(float, op_desc.GetAttr("value"));
        str_value = std::to_string(value);
      }
      auto* dev_ctx = static_cast<phi::CPUContext*>(
          platform::DeviceContextPool::Instance().Get(platform::CPUPlace()));
      std::unique_ptr<phi::DenseTensor> out_tensor(new phi::DenseTensor());
      out_tensor->Resize(phi::make_ddim(shape));
      nvinfer1::DataType trt_dtype = nvinfer1::DataType::kFLOAT;
      void* trt_data = nullptr;
      size_t trt_num;
      if (dtype == 2 || dtype == 3) {  // int,int64
        auto* tmp_ptr = dev_ctx->template HostAlloc<int>(out_tensor.get());
        for (int64_t i = 0; i < out_tensor->numel(); i++)
          tmp_ptr[i] = std::stoi(str_value);
        trt_dtype = nvinfer1::DataType::kINT32;
        trt_data = static_cast<void*>(tmp_ptr);
      } else if (dtype == 5) {  // float
        auto* tmp_ptr = dev_ctx->template HostAlloc<float>(out_tensor.get());
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
      nvinfer1::ILayer* layer =
          TRT_ENGINE_ADD_LAYER(engine_, Constant, trt_in_shape, weight.get());
      auto output_name = op_desc.Output("Out")[0];
      RreplenishLayerAndOutput(
          layer, "fill_constant", {output_name}, test_mode);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fill_constant, FillConstantOpConverter);
