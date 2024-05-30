/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

/*
 * Einsum Op
 */
class EinsumOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
#if IS_TRT_VERSION_GE(8200)
    VLOG(3) << "convert a einsum op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);
    auto operand_inputs = op_desc.Input("Operands");
    auto equation = PADDLE_GET_CONST(std::string, op_desc.GetAttr("equation"));
    std::vector<nvinfer1::ITensor*> input_tensors;
    for (auto input_name : operand_inputs) {
      auto tmp_tensor = engine_->GetITensor(input_name);
      input_tensors.push_back(tmp_tensor);
    }

    int32_t input_num = static_cast<int32_t>(operand_inputs.size());
    auto layer = TRT_ENGINE_ADD_LAYER(
        engine_, Einsum, input_tensors.data(), input_num, equation.c_str());

    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "einsum", {output_name}, test_mode);
#else
    VLOG(3) << "Einsum is not supported when TensorRT < 8.2.0";
#endif
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(einsum, EinsumOpConverter);
