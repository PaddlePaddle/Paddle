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
#include "paddle/fluid/inference/tensorrt/plugin/prelu_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * PRelu converter from fluid to tensorRT.
 */
class PReluOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert fluid prelu op to tensorrt prelu layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    // Get attrs
    std::string mode = PADDLE_GET_CONST(std::string, op_desc.GetAttr("mode"));
    std::string data_format = "NCHW";
    if (op_desc.HasAttr("data_format")) {
      data_format =
          PADDLE_GET_CONST(std::string, op_desc.GetAttr("data_format"));
    }
    auto* alpha_tensor = engine_->GetITensor(op_desc.Input("Alpha")[0]);

    auto alpha_dims = alpha_tensor->getDimensions();
    auto input_dims = input->getDimensions();
    nvinfer1::ITensor* real_alpha_tensor = alpha_tensor;
    if (alpha_dims.nbDims == 1 && alpha_dims.nbDims != input_dims.nbDims) {
      auto* reshape_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *alpha_tensor);
      int c = alpha_dims.d[0];
      if (engine_->with_dynamic_shape()) {
        auto* n_tensor = Add1DConstantLayer(1);
        auto* c_tensor = Add1DConstantLayer(c);
        auto* hw_tensor = Add1DConstantLayer(std::vector<int32_t>{1, 1});
        if (data_format == "NCHW") {
          auto* shape_tensor = Concat(
              std::vector<nvinfer1::ITensor*>{n_tensor, c_tensor, hw_tensor});
          reshape_layer->setInput(1, *shape_tensor);
        } else {
          auto* shape_tensor = Concat(
              std::vector<nvinfer1::ITensor*>{n_tensor, hw_tensor, c_tensor});
          reshape_layer->setInput(1, *shape_tensor);
        }
      } else {
        nvinfer1::Dims3 reshape_dim{1, 1, 1};
        if (data_format == "NCHW") {
          reshape_dim.d[0] = c;
        } else if (data_format == "NHWC") {
          reshape_dim.d[input_dims.nbDims - 1] = c;
        }
        reshape_layer->setReshapeDimensions(reshape_dim);
      }
      real_alpha_tensor = reshape_layer->getOutput(0);
    }

    nvinfer1::ILayer* layer = nullptr;

    layer = TRT_ENGINE_ADD_LAYER(
        engine_, ParametricReLU, *input, *real_alpha_tensor);

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "prelu", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(prelu, PReluOpConverter);
