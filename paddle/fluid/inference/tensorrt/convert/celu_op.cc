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

class CeluOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert celu op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    int input_num = op_desc.Input("X").size();
    PADDLE_ENFORCE_EQ(input_num,
                      1,
                      common::errors::InvalidArgument(
                          "The input X's size must equal to 1 in TRT celu op."
                          " But received X's size %d.",
                          input_num));
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    // Get output
    size_t output_num = op_desc.Output("Out").size();
    PADDLE_ENFORCE_EQ(
        output_num,
        1UL,
        common::errors::InvalidArgument(
            "The output Out's size must equal to 1 in TRT celu op. "
            "But received Out's size %u.",
            output_num));
    // Get attrs
    float alpha = PADDLE_GET_CONST(float, op_desc.GetAttr("alpha"));

    nvinfer1::ILayer* layer = nullptr;

    int32_t rank = input->getDimensions().nbDims;
    nvinfer1::Dims constant_shape;
    constant_shape.nbDims = rank;
    std::fill(constant_shape.d, constant_shape.d + rank, 1);
    std::vector<float> weight_alpha_data{alpha};
    std::vector<float> weight_zero_data{0.f};
    std::vector<float> weight_one_data{1.f};
    auto* alpha_data =
        AddConstantLayer(weight_alpha_data.data(), constant_shape);
    auto* constant_zero_data =
        AddConstantLayer(weight_zero_data.data(), constant_shape);
    auto* constant_one_data =
        AddConstantLayer(weight_one_data.data(), constant_shape);

    auto* input_div_with_alpha = Div(input, alpha_data);
    auto* input_exp = TRT_ENGINE_ADD_LAYER(
        engine_, Unary, *input_div_with_alpha, nvinfer1::UnaryOperation::kEXP);
    auto* input_sub_with_one = Sub(input_exp->getOutput(0), constant_one_data);
    auto* input_prod_with_alpha = Prod(input_sub_with_one, alpha_data);
    auto* min_input = Min(input_prod_with_alpha, constant_zero_data);
    auto* relu = TRT_ENGINE_ADD_LAYER(
        engine_, Activation, *input, nvinfer1::ActivationType::kRELU);
    layer = TRT_ENGINE_ADD_LAYER(engine_,
                                 ElementWise,
                                 *relu->getOutput(0),
                                 *min_input,
                                 nvinfer1::ElementWiseOperation::kSUM);

    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "celu", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(celu, CeluOpConverter);
