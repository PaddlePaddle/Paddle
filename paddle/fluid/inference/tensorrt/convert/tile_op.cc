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

class TileOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
#if IS_TRT_VERSION_GE(7000)
    VLOG(3) << "convert a tile op to tensorrt tile layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto inputs = op_desc.Inputs();
    auto input_shape = input->getDimensions();
    auto rank = input_shape.nbDims;
    auto output_name = op_desc.Output("Out")[0];

    auto input_shape_tensor = Shape(input);

    nvinfer1::ITensor* repeat_tensor = nullptr;
    int32_t repeat_rank = 0;
    if (inputs.find("RepeatTimes") != inputs.end() &&
        !op_desc.Input("RepeatTimes").empty()) {
      repeat_tensor = engine_->GetITensor(op_desc.Input("RepeatTimes")[0]);
      repeat_rank = repeat_tensor->getDimensions().d[0];
    } else if (inputs.find("repeat_times_tensor") != inputs.end() &&
               !op_desc.Input("repeat_times_tensor").empty()) {
      int32_t repeat_size = op_desc.Input("repeat_times_tensor").size();
      std::vector<nvinfer1::ITensor*> repeat_tensors;
      for (int32_t i = 0; i < repeat_size; ++i) {
        repeat_tensors.push_back(
            engine_->GetITensor(op_desc.Input("repeat_times_tensor")[i]));
      }
      repeat_tensor = Concat(repeat_tensors);
      repeat_rank = repeat_size;
    } else {
      std::vector<int32_t> repeat_times = PADDLE_GET_CONST(
          std::vector<int32_t>, op_desc.GetAttr("repeat_times"));
      repeat_tensor =
          Add1DConstantLayer(repeat_times, output_name + "_shape_tensor_");
      repeat_rank = repeat_times.size();
    }

    nvinfer1::ITensor* repeat_expand_tensor;
    if (rank > repeat_rank) {
      auto* one_rank_tensor =
          Add1DConstantLayer(std::vector<int32_t>(rank - repeat_rank, 1),
                             output_name + "_one_rank_tensor_");
      std::vector<nvinfer1::ITensor*> itensors;
      itensors.push_back(one_rank_tensor);
      itensors.push_back(repeat_tensor);
      repeat_expand_tensor = Concat(itensors);
    }
    if (rank < repeat_rank) {
      auto* one_rank_tensor =
          Add1DConstantLayer(std::vector<int32_t>(repeat_rank - rank, 1));
      std::vector<nvinfer1::ITensor*> itensors;
      itensors.push_back(one_rank_tensor);
      itensors.push_back(input_shape_tensor);
      input_shape_tensor = Concat(itensors);
      // need reshape input to more dims.
      input = Reshape(input, input_shape_tensor, "reshape_input_before_slice");
      repeat_expand_tensor = repeat_tensor;
    } else {
      repeat_expand_tensor = repeat_tensor;
    }
    std::vector<int32_t> start(std::max(rank, repeat_rank), 0);
    std::vector<int32_t> stride(std::max(rank, repeat_rank), 1);
    auto start_tensor = Add1DConstantLayer(start, output_name + "start_tensor");
    auto stride_tensor =
        Add1DConstantLayer(stride, output_name + "stride_tensor");
    auto output_shape_tensor = Prod(input_shape_tensor, repeat_expand_tensor);
    auto layer = TRT_ENGINE_ADD_LAYER(engine_,
                                      Slice,
                                      *input,
                                      nvinfer1::Dims{},
                                      nvinfer1::Dims{},
                                      nvinfer1::Dims{});

    layer->setInput(1, *start_tensor);
    layer->setInput(2, *output_shape_tensor);
    layer->setInput(3, *stride_tensor);
#if IS_TRT_VERSION_GE(8600)
    layer->setMode(nvinfer1::SampleMode::kWRAP);
#else
    layer->setMode(nvinfer1::SliceMode::kWRAP);
#endif
    ReplenishLayerAndOutput(layer, "tile", {output_name}, test_mode);
#endif
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(tile, TileOpConverter);
