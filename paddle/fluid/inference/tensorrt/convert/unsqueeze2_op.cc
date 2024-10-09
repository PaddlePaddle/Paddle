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

class Unsqueeze2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a unsqueeze2 op to tensorrt shuffle layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto input_dims = input->getDimensions();
    auto output_name = op_desc.Output("Out")[0];

    // Get Attrs
    std::vector<int> axes =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("axes"));
    PADDLE_ENFORCE_GT(
        axes.size(),
        0,
        common::errors::InvalidArgument(
            "Attr(axes).size should be > 0 in unsqueeze2 op in TensorRT,"
            "but received axes.size() = %d.",
            axes.size()));

    std::vector<bool> should_unsqueeze(input_dims.nbDims + axes.size(), false);
    int cur_out_rank = input_dims.nbDims;
    for (size_t i = 0; i < axes.size(); i++) {
      cur_out_rank++;
      axes[i] += (axes[i] < 0) ? cur_out_rank : 0;
      // axes[i] is relative to cur_out_rank
      // we make [axes[i], cur_out_rank - 2] shift right
      // and make (axes[i]) to true!
      for (int j = cur_out_rank - 1; j > axes[i]; j--) {
        should_unsqueeze[j] = should_unsqueeze[j - 1];
      }
      if (axes[i] >= cur_out_rank)
        should_unsqueeze[cur_out_rank - 1] = true;
      else
        should_unsqueeze[axes[i]] = true;
    }

    std::vector<int32_t> gather_indices;
    int in_rank_i = 0;
    for (size_t i = 0; i < should_unsqueeze.size(); i++) {
      if (should_unsqueeze[i]) {
        gather_indices.push_back(input_dims.nbDims);
        continue;
      }
      gather_indices.push_back(in_rank_i);
      in_rank_i++;
    }

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
    auto* shape_tensor = Shape(input);
    std::vector<int32_t> all_one(axes.size(), 1);
    auto* all_one_tensor = Add1DConstantLayer(all_one);
    std::vector<nvinfer1::ITensor*> concat_inputs = {shape_tensor,
                                                     all_one_tensor};
    auto* real_shape_tensor = Gather(Concat(concat_inputs), gather_indices);
    layer->setInput(1, *real_shape_tensor);
    ReplenishLayerAndOutput(layer, "unsqueeze2", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(unsqueeze2, Unsqueeze2OpConverter);
