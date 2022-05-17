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

class Unsqueeze2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert a fluid unsqueeze2 op to tensorrt shuffle layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto input_dims = input->getDimensions();
    auto output_name = op_desc.Output("Out")[0];

    // Get Attrs
    std::vector<int> axes =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("axes"));
    PADDLE_ENFORCE_GT(
        axes.size(), 0,
        platform::errors::InvalidArgument(
            "Attr(axes).size should be > 0 in unsqueeze2 op in TensorRT,"
            "but received axes.size() = %d.",
            axes.size()));

    std::vector<bool> should_unsqueeze(input_dims.nbDims + axes.size(), false);
    for (size_t i = 0; i < axes.size(); i++) {
      if (engine_->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(6000)
        axes[i] += (axes[i] < 0) ? should_unsqueeze.size() : 0;
#endif
      } else {
        axes[i] += (axes[i] < 0) ? should_unsqueeze.size() : -1;
      }
      should_unsqueeze[axes[i]] = true;
    }

    nvinfer1::Dims trt_out_dims;
    trt_out_dims.nbDims = should_unsqueeze.size();
    std::vector<int32_t> gather_indices;
    int in_rank_i = 0;
    for (size_t i = 0; i < should_unsqueeze.size(); i++) {
      if (should_unsqueeze[i]) {
        trt_out_dims.d[i] = 1;
        gather_indices.push_back(input_dims.nbDims);
        continue;
      }
      trt_out_dims.d[i] = input_dims.d[in_rank_i];
      gather_indices.push_back(in_rank_i);
      in_rank_i++;
    }
    std::string name = "_add_unsqueeze2_op_";
    auto gather_indices_tensor =
        Add1DConstantLayer(gather_indices, name + "gather_indices");
    std::vector<int32_t> all_one(axes.size(), 1);
    auto all_one_tensor = Add1DConstantLayer(all_one, name + "all_one");

    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      auto shape_tensor =
          TRT_ENGINE_ADD_LAYER(engine_, Shape, *input)->getOutput(0);
      std::vector<nvinfer1::ITensor*> concat_inputs = {shape_tensor,
                                                       all_one_tensor};
      auto real_shape_tensor =
          TRT_ENGINE_ADD_LAYER(engine_, Gather,
                               *TRT_ENGINE_ADD_LAYER(engine_, Concatenation,
                                                     concat_inputs.data(), 2)
                                    ->getOutput(0),
                               *gather_indices_tensor, 0)
              ->getOutput(0);

      layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      layer->setInput(1, *real_shape_tensor);

    } else {
      auto unsqueeze_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      unsqueeze_layer->setReshapeDimensions(trt_out_dims);
      layer = dynamic_cast<nvinfer1::ILayer*>(unsqueeze_layer);
    }
    RreplenishLayerAndOutput(layer, "unsqueeze2", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(unsqueeze2, Unsqueeze2OpConverter);
