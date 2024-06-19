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
#include "paddle/fluid/inference/tensorrt/helper.h"

namespace paddle::inference::tensorrt {
/*
 * Stack converter from fluid to tensorRT.
 */
class RollOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert roll op to tensorrt Gather layer";

    framework::OpDesc op_desc(op, nullptr);
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    std::vector<int64_t> axis =
        PADDLE_GET_CONST(std::vector<int64_t>, op_desc.GetAttr("axis"));
    std::vector<int64_t> shifts =
        PADDLE_GET_CONST(std::vector<int64_t>, op_desc.GetAttr("shifts"));
    int axis_size = axis.size();
    nvinfer1::ITensor* input_shape_tensor = Shape(input);
    nvinfer1::ILayer* layer = nullptr;
    for (int i = 0; i < axis_size; i++) {
      auto axi = static_cast<int32_t>(axis[i]);
      auto shift = static_cast<int32_t>(shifts[i]);
      nvinfer1::ITensor* input_axis =
          GetEleTensorOfShape(input_shape_tensor, axi);
      nvinfer1::ITensor* input_shift = Add1DConstantLayer(shift);
      // 1.sub_value mod input_axis
      auto input1 = Sub(input_axis, input_shift);
      auto tmp_div_res = FloorDiv(input1, input_axis);
      auto tmp_prod_res = Prod(tmp_div_res, input_axis);
      auto start = Sub(input1, tmp_prod_res);
      // 2.avoid start less than 0,start mod input_axis
      start = Sum(start, input_axis);
      auto tmp_div_res1 = FloorDiv(start, input_axis);
      auto tmp_prod_res1 = Prod(tmp_div_res1, input_axis);
      start = Sub(start, tmp_prod_res1);

      auto zero_tensor = Add1DConstantLayer(0);
      auto step = Add1DConstantLayer(1);
      // 3.make index_tensor0
      auto quotient_tensor = FloorDiv(Sub(input_axis, start), step);
      auto* start1 = GetEleTensorOfShape(start, 0, true);
      auto fill_layer0 = TRT_ENGINE_ADD_LAYER(
          engine_, Fill, nvinfer1::Dims{}, nvinfer1::FillOperation::kLINSPACE);
      fill_layer0->setInput(0, *quotient_tensor);
      fill_layer0->setInput(1, *start1);
      fill_layer0->setInput(2, *step);
      auto* index_tensor0 = fill_layer0->getOutput(0);
      // 4.make index_tensor1
      quotient_tensor = FloorDiv(Sub(start, zero_tensor), step);
      auto* start2 = Add1DConstantLayer(0, "", true);
      auto fill_layer1 = TRT_ENGINE_ADD_LAYER(
          engine_, Fill, nvinfer1::Dims{}, nvinfer1::FillOperation::kLINSPACE);
      fill_layer1->setInput(0, *quotient_tensor);
      fill_layer1->setInput(1, *start2);
      fill_layer1->setInput(2, *step);
      auto* index_tensor1 = fill_layer1->getOutput(0);
      std::vector<nvinfer1::ITensor*> itensors;
      itensors.push_back(index_tensor0);
      itensors.push_back(index_tensor1);
      nvinfer1::ITensor* concat_input_tensor = Concat(itensors);
      if (layer == nullptr) {
        layer = TRT_ENGINE_ADD_LAYER(
            engine_, Gather, *input, *concat_input_tensor, axi);
      } else {
        layer = TRT_ENGINE_ADD_LAYER(
            engine_, Gather, *layer->getOutput(0), *concat_input_tensor, axi);
      }
    }
    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "roll", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(roll, RollOpConverter);
