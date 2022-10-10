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

class DequantizeLinearOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a dequantize_linear op to tensorrt IDequantizeLayer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* x = engine_->GetITensor(op_desc.Input("X")[0]);
    std::cout << op_desc.Input("X")[0] << std::endl;
    auto* scale_var = scope.FindVar(op_desc.Input("Scale")[0]);
    PADDLE_ENFORCE_NOT_NULL(
        scale_var,
        platform::errors::NotFound("Can not find %s presistale var in scope.",
                                   op_desc.Input("Scale")[0]));
    auto* scale_t = scale_var->GetMutable<framework::LoDTensor>();
    const float* fp32_data = reinterpret_cast<const float*>(
        engine_->GetTrtWeight(op_desc.Input("Scale")[0], *scale_t)
            .get()
            .values);
    std::vector<float> new_fp32_data(scale_t->numel(), 0);
    for (int i = 0; i < scale_t->numel(); i++) {
      new_fp32_data[i] = fp32_data[i] / 127.;
    }
    nvinfer1::Dims a;
    a.nbDims = 1;
    a.d[0] = scale_t->numel();
    auto* scale = AddConstantLayer(new_fp32_data.data(), a);

    int axis = PADDLE_GET_CONST(int, op_desc.GetAttr("quant_axis"));
    if (axis == -1) {
      axis = 1;
    }
    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Dequantize, *x, *scale);
    layer->setAxis(axis);
    auto output_name = op_desc.Output("Y")[0];
    RreplenishLayerAndOutput(
        layer, "dequantize_linear", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(dequantize_linear, DequantizeLinearOpConverter);
