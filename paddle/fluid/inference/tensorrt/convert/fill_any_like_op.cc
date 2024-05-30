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

namespace paddle::inference::tensorrt {

class FillAnyLikeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert fill_any_like op to tensorrt layer ";
    framework::OpDesc op_desc(op, nullptr);
    auto* input = engine_->GetITensor(op_desc.Input("X").front());
    auto output_name = op_desc.Output("Out").front();
    auto input_dims = input->getDimensions();
    auto nbDims_num = input_dims.nbDims;
    nvinfer1::ITensor* value_tensor;

    const int dtype = PADDLE_GET_CONST(int, op_desc.GetAttr("dtype"));
    float value = PADDLE_GET_CONST(float, op_desc.GetAttr("value"));
    if ((dtype == 2) ||
        (dtype == -1 && input->getType() == nvinfer1::DataType::kINT32)) {
      value_tensor = Add1DConstantLayer(static_cast<int32_t>(value),
                                        output_name + "_value_tensor_");
    } else if (dtype == 3) {
      LOG(WARNING) << "the fill_any_like has int64 dtype, it "
                      "will be cast to int32.";
      value_tensor = Add1DConstantLayer(static_cast<int32_t>(value),
                                        output_name + "_value_tensor_");
    } else {
      value_tensor = Add1DConstantLayer(value, output_name + "_value_tensor_");
    }
    auto shape_tensor = Shape(input);
    auto* one_rank_tensor = Add1DConstantLayer(
        std::vector<int32_t>(nbDims_num, 1), output_name + "_one_rank_tensor_");
    auto input_shape_tensor = one_rank_tensor;
    auto* shuffle = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *value_tensor);
    shuffle->setInput(1, *input_shape_tensor);

    std::vector<int32_t> start_vec(nbDims_num, 0);
    nvinfer1::Dims start;
    start.nbDims = nbDims_num;
    for (int32_t i = 0; i < nbDims_num; ++i) {
      start.d[i] = start_vec[i];
    }
    nvinfer1::Dims size;
    size.nbDims = nbDims_num;
    nvinfer1::Dims stride;
    stride.nbDims = nbDims_num;

    auto starts_tensor =
        Add1DConstantLayer(start_vec, output_name + "_start_tensor_");
    auto one_tensor = Add1DConstantLayer(1, output_name + "_one_tensor_");

    auto sizes_tensor = Max(input_shape_tensor, shape_tensor);
    auto input_sub_tensor = Sub(input_shape_tensor, one_tensor);
    auto strides_tensor = Min(one_tensor, input_sub_tensor);

    auto layer = TRT_ENGINE_ADD_LAYER(
        engine_, Slice, *shuffle->getOutput(0), start, size, stride);
    layer->setInput(1, *starts_tensor);
    layer->setInput(2, *sizes_tensor);
    layer->setInput(3, *strides_tensor);

    ReplenishLayerAndOutput(layer, "fill_any_like", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(fill_any_like, FillAnyLikeOpConverter);
