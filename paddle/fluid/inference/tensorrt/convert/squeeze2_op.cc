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

class Squeeze2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert a fluid squeeze2 op to tensorrt shuffle layer";

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
            "Attr(axes).size should be > 0 in squeeze2 op in TensorRT,"
            "but received axes.size() = %d.",
            axes.size()));

    std::vector<bool> should_squeeze(input_dims.nbDims, false);
    for (size_t i = 0; i < axes.size(); i++) {
      if (engine_->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(6000)
        axes[i] += (axes[i] < 0) ? input_dims.nbDims : 0;
#endif
      } else {
        axes[i] += (axes[i] < 0) ? input_dims.nbDims : -1;
      }
      if (input_dims.d[axes[i]] == 1) {
        should_squeeze[axes[i]] = true;
      }
    }

    nvinfer1::Dims trt_out_dims;
    trt_out_dims.nbDims = 0;
    std::vector<int32_t> gather_indices;
    for (size_t i = 0; i < should_squeeze.size(); i++) {
      if (should_squeeze[i]) continue;
      trt_out_dims.nbDims++;
      trt_out_dims.d[i] = input_dims.d[i];
      gather_indices.push_back(i);
    }
    std::string name = "_add_squeeze2_op_";
    auto gather_indices_tensor =
        Add1DConstantLayer(gather_indices, name + "gather_indices");
    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      auto shape_tensor =
          TRT_ENGINE_ADD_LAYER(engine_, Shape, *input)->getOutput(0);
      auto real_shape_tensor =
          TRT_ENGINE_ADD_LAYER(engine_, Gather, *shape_tensor,
                               *gather_indices_tensor, 0)
              ->getOutput(0);
      layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      layer->setInput(1, *real_shape_tensor);

    } else {
      auto squeeze_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      squeeze_layer->setReshapeDimensions(trt_out_dims);
      layer = dynamic_cast<nvinfer1::ILayer*>(squeeze_layer);
    }
    RreplenishLayerAndOutput(layer, "squeeze2", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(squeeze2, Squeeze2OpConverter);
