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

namespace paddle {
namespace inference {
namespace tensorrt {

class RangeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a range op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);
    nvinfer1::ILayer* layer = nullptr;
    nvinfer1::ITensor* quotient_tensor;
    nvinfer1::ITensor* end;
    bool is_shape_tensor = true;

    // Declare inputs
    auto* start = engine_->GetITensor(op_desc.Input("Start")[0]);
    auto* step = engine_->GetITensor(op_desc.Input("Step")[0]);
    auto output_name = op_desc.Output("Out")[0];
    auto end_name = op_desc.Input("End")[0];

    auto min_shape = engine_->min_shape_tensor();
    auto max_shape = engine_->max_shape_tensor();
    auto optim_shape = engine_->optim_shape_tensor();

    if (start->getType() != nvinfer1::DataType::kFLOAT) {
      // Check whether the specific value of Input End has been collected in the
      // file "shape_range.pbtxt". If so, convert Input End to a ConstantTensor.
      while (min_shape.count(end_name) == 0 && is_shape_tensor) {
        if (!end_name.empty()) {
          end_name.pop_back();
        } else {
          is_shape_tensor = false;
        }
      }
      // Check that the collected min, max, opt values are equal.
      if (is_shape_tensor && min_shape[end_name][0] == max_shape[end_name][0] &&
          min_shape[end_name][0] == optim_shape[end_name][0]) {
        end = Add1DConstantLayer(min_shape[end_name][0],
                                 output_name + "_end_tensor_");
      } else {
        end = engine_->GetITensor(op_desc.Input("End")[0]);
      }
    } else {
      end = engine_->GetITensor(op_desc.Input("End")[0]);
    }

    auto zero_tensor = Add1DConstantLayer(0, output_name + "_zero_tensor_");
    auto fquotient_tensor = FloorDiv(Sub(start, end), step);
    if (start->getType() == nvinfer1::DataType::kFLOAT) {
      auto* cast_int32_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Identity, *fquotient_tensor);
      cast_int32_layer->setOutputType(0, nvinfer1::DataType::kINT32);
      cast_int32_layer->getOutput(0)->setType(nvinfer1::DataType::kINT32);
      quotient_tensor = cast_int32_layer->getOutput(0);
    } else {
      quotient_tensor = fquotient_tensor;
    }
    auto number_tensor = Max(Sub(zero_tensor, quotient_tensor), zero_tensor);
    auto* start1 = engine_->GetITensor(op_desc.Input("Start")[0], true);

    layer = TRT_ENGINE_ADD_LAYER(
        engine_, Fill, nvinfer1::Dims{}, nvinfer1::FillOperation::kLINSPACE);
    layer->setInput(0, *number_tensor);
    layer->setInput(1, *start1);
    layer->setInput(2, *step);

    RreplenishLayerAndOutput(layer, "range", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(range, RangeOpConverter);
