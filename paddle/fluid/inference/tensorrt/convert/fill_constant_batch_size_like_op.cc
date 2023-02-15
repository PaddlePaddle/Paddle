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

class FillConstantBatchSizeLikeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
#if IS_TRT_VERSION_GE(7000)
    VLOG(4) << "convert a fluid fill_constant_batch_size_like op to tensorrt "
               "fill_constant_batch_size_like layer";

    framework::OpDesc op_desc(op, nullptr);
    auto* input = engine_->GetITensor(op_desc.Input("Input")[0]);
    int dtype = PADDLE_GET_CONST(int, op_desc.GetAttr("dtype"));
    // be float
    PADDLE_ENFORCE_EQ(dtype,
                      5,
                      platform::errors::InvalidArgument(
                          "fill_constant_batch_size_like's input data type "
                          "must be float in Paddle-TRT."));

    int input_dim_idx = PADDLE_GET_CONST(int, op_desc.GetAttr("input_dim_idx"));
    size_t output_dim_idx =
        PADDLE_GET_CONST(int, op_desc.GetAttr("output_dim_idx"));
    std::string str_value =
        PADDLE_GET_CONST(std::string, op_desc.GetAttr("str_value"));
    std::vector<int32_t> shape =
        PADDLE_GET_CONST(std::vector<int32_t>, op_desc.GetAttr("shape"));
    if (str_value == "") {
      float value = PADDLE_GET_CONST(float, op_desc.GetAttr("value"));
      str_value = std::to_string(value);
    }
    float value = std::stof(str_value);

    auto* input_shape_tensor = Shape(input);
    auto* batch_tensor = GetEleTensorOfShape(input_shape_tensor, input_dim_idx);
    std::string name = "_add_fill_constant_batch_size_like_op_";
    auto shape_attr_tensor = Add1DConstantLayer(shape, name + "shape_attr");
    std::vector<int32_t> gather_out_shape_indices;
    for (size_t i = 0; i < shape.size(); i++) {
      if (i == output_dim_idx) {
        gather_out_shape_indices.push_back(shape.size());
        continue;
      }
      gather_out_shape_indices.push_back(i);
    }
    std::vector<nvinfer1::ITensor*> concat_inputs{shape_attr_tensor,
                                                  batch_tensor};
    auto out_shape_tensor =
        Gather(Concat(concat_inputs), gather_out_shape_indices);
    auto layer = TRT_ENGINE_ADD_LAYER(
        engine_, Fill, nvinfer1::Dims{}, nvinfer1::FillOperation::kLINSPACE);
    std::vector<float> value_vec(1, value);
    std::vector<float> beta_vec(shape.size(), 0.);
    layer->setAlpha(value);
    layer->setBeta(0.f);
    layer->setInput(0, *out_shape_tensor);
    layer->setInput(1, *Add1DConstantLayer(value_vec, name + "alpha", true));
    layer->setInput(2, *Add1DConstantLayer(beta_vec, name + "beta", false));
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(
        layer, "fill_constant_batch_size_like", {output_name}, test_mode);
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fill_constant_batch_size_like,
                          FillConstantBatchSizeLikeOpConverter);
