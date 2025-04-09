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
#include "paddle/fluid/inference/tensorrt/plugin/split_op_plugin.h"

namespace paddle::inference::tensorrt {

class SplitOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a split op to tensorrt split layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto inputs = op_desc.Inputs();
    auto input_dims = input->getDimensions();
    int output_num = op_desc.Output("Out").size();

    // Get Attrs
    int axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
    int num = 0;
    std::vector<int> output_lengths =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("sections"));
    if (op_desc.HasAttr("num")) {
      num = PADDLE_GET_CONST(int, op_desc.GetAttr("num"));
    }
    nvinfer1::ITensor* shape_tensor = nullptr;
    if (engine_->with_dynamic_shape()) {
      axis += (axis < 0) ? input_dims.nbDims : 0;
      // only be called in dynamic_shape mode
      shape_tensor = Shape(input);
    } else {
      axis += (axis < 0) ? input_dims.nbDims : -1;
    }
    bool in_axis_dim_dynamic = false;
    bool sections_tensor_list = false;
    nvinfer1::ITensor* sections_tensor = nullptr;

    // need infer output_lengths
    if (inputs.find("SectionsTensorList") != inputs.end() &&
        !op_desc.Input("SectionsTensorList").empty()) {
      int32_t sections_size = op_desc.Input("SectionsTensorList").size();
      std::vector<nvinfer1::ITensor*> sections_tensors;
      for (int32_t i = 0; i < sections_size; ++i) {
        sections_tensors.push_back(
            engine_->GetITensor(op_desc.Input("SectionsTensorList")[i]));
      }
      sections_tensor = Concat(sections_tensors);
      sections_tensor_list = true;
    } else if (!output_lengths.empty()) {
      sections_tensor = Add1DConstantLayer(output_lengths);
    } else if (num > 0 && output_lengths.empty()) {
      if (input_dims.d[axis] > 0) {
        int64_t in_axis_dim = input_dims.d[axis];
        size_t out_axis_dim = in_axis_dim / num;
        for (int i = 0; i < num; ++i) {
          output_lengths.push_back(out_axis_dim);
        }
        sections_tensor = Add1DConstantLayer(output_lengths);
      } else {
        in_axis_dim_dynamic = true;
        auto* num_tensor = Add1DConstantLayer(num);
        sections_tensor =
            Div(GetEleTensorOfShape(shape_tensor, axis), num_tensor);
      }
    }

    nvinfer1::ILayer* layer = nullptr;
    nvinfer1::Dims trt_step_dims;
    trt_step_dims.nbDims = input->getDimensions().nbDims;
    for (int i = 0; i < trt_step_dims.nbDims; i++) trt_step_dims.d[i] = 1;

    std::vector<int32_t> gather_indices;
    gather_indices.resize(trt_step_dims.nbDims);
    std::iota(gather_indices.begin(), gather_indices.end(), 0);
    gather_indices[axis] = gather_indices.size();
    std::vector<int32_t> zeros(trt_step_dims.nbDims, 0);
    std::vector<int32_t> stride(trt_step_dims.nbDims, 1);
    auto zeros_tensor = Add1DConstantLayer(zeros);
    auto stride_tensor = Add1DConstantLayer(stride);
    // input : [N,C,H,W]
    nvinfer1::ITensor* start_point_tensor = zeros_tensor;
    nvinfer1::ITensor* this_len_tensor = zeros_tensor;
    for (int i = 0; i < output_num; i++) {
      if (sections_tensor_list || !in_axis_dim_dynamic) {
        start_point_tensor = Sum(start_point_tensor, this_len_tensor);
        this_len_tensor = Gather(sections_tensor, std::vector<int32_t>{i});
      } else {
        this_len_tensor = sections_tensor;
        auto* i_tensor = Add1DConstantLayer(static_cast<int>(i));
        start_point_tensor = Prod(i_tensor, sections_tensor);
      }

      std::vector<nvinfer1::ITensor*> concat_inputs1 = {zeros_tensor,
                                                        start_point_tensor};
      std::vector<nvinfer1::ITensor*> concat_inputs2 = {shape_tensor,
                                                        this_len_tensor};
      auto* start_tensor = Gather(Concat(concat_inputs1), gather_indices);
      auto* size_tensor = Gather(Concat(concat_inputs2), gather_indices);
      layer = TRT_ENGINE_ADD_LAYER(engine_,
                                   Slice,
                                   *input,
                                   nvinfer1::Dims{},
                                   nvinfer1::Dims{},
                                   nvinfer1::Dims{});
      layer->setInput(1, *start_tensor);
      layer->setInput(2, *size_tensor);
      layer->setInput(3, *stride_tensor);

      auto output_name = op_desc.Output("Out")[i];
      ReplenishLayerAndOutput(layer, "split", {output_name}, test_mode);
    }
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(split, SplitOpConverter);
