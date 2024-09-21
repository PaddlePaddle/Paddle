/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
 * TransposeOp
 */
class TransposeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a transpose op to tensorrt shuffle layer";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    int dims = input->getDimensions().nbDims;
    std::vector<int> axis =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("axis"));
    nvinfer1::Permutation perm;
    for (int i = 0; i < dims; i++) {
      int j = i;
      perm.order[i] = axis[j];
    }
    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    layer->setFirstTranspose(perm);

    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "transpose", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(transpose, TransposeOpConverter);
REGISTER_TRT_OP_CONVERTER(transpose2, TransposeOpConverter);
