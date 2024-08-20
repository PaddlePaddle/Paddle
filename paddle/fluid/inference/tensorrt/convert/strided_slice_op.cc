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

class StridedSliceOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert strided_slice op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("Input")[0]);
    auto output_name = op_desc.Output("Out")[0];

    // phi only allow axes[i] >= 0 && <rank, so we need not deal with minus
    // axes[i]
    std::vector<int> axes =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("axes"));
    std::vector<int> starts =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("starts"));
    std::vector<int> ends =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("ends"));
    std::vector<int> strides =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));
    std::vector<int> decrease_axises =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("decrease_axis"));

    auto input_dims = input->getDimensions();

    auto* size_tensor =
        Sub(start_tensor, Min(Concat(end_vec_tensor), shape_tensor));
    auto zero_t =
        Add1DConstantLayer(std::vector<int>(nchw_input_dims.nbDims, 0));
    auto step_tensor = Add1DConstantLayer(trt_step_dims);
    size_tensor = Sub(zero_t, FloorDiv(size_tensor, step_tensor));

    layer = TRT_ENGINE_ADD_LAYER(
        engine_, Slice, *input, trt_start_dims, trt_size_dims, trt_step_dims);
    layer->setInput(1, *start_tensor);
    layer->setInput(2, *size_tensor);
    layer->setInput(3, *step_tensor);

    if (!decrease_axises.empty()) {
      std::vector<int32_t> gather_indices;
      for (int i = 0; i < trt_size_dims.nbDims; i++) {
        if (decrease_axises.end() !=
            std::find(decrease_axises.begin(), decrease_axises.end(), i))
          continue;
        gather_indices.push_back(i);
      }
      if (gather_indices.empty()) gather_indices.push_back(decrease_axises[0]);
      auto real_size_tensor = Gather(size_tensor, gather_indices);
      layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *layer->getOutput(0));
      layer->setInput(1, *real_size_tensor);
    }
    ReplenishLayerAndOutput(layer, "strided_slice", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(strided_slice, StridedSliceOpConverter);
