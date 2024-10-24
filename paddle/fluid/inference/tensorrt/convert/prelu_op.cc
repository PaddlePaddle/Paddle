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

/*
 * PRelu converter from paddle to tensorRT.
 */
class PReluOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert prelu op to tensorrt prelu layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto input_dims = input->getDimensions();
    // Get attrs
    std::string mode = PADDLE_GET_CONST(std::string, op_desc.GetAttr("mode"));
    std::string data_format = "NCHW";
    if (op_desc.HasAttr("data_format")) {
      data_format =
          PADDLE_GET_CONST(std::string, op_desc.GetAttr("data_format"));
    }

    auto* alpha_var = scope.FindVar(op_desc.Input("Alpha")[0]);
    auto* alpha_weight = alpha_var->GetMutable<phi::DenseTensor>();
    auto w_dims = alpha_weight->dims();
    auto alpha_data =
        engine_->GetFp32TrtWeight(op_desc.Input("Alpha")[0], *alpha_weight);

    nvinfer1::Dims trt_w_dims;
    trt_w_dims.nbDims = w_dims.size();
    for (int i = 0; i < trt_w_dims.nbDims; i++) {
      trt_w_dims.d[i] = w_dims[i];
    }

    nvinfer1::ITensor* alpha_tensor =
        TRT_ENGINE_ADD_LAYER(engine_, Constant, trt_w_dims, alpha_data.get())
            ->getOutput(0);

    auto alpha_dims = alpha_tensor->getDimensions();
    nvinfer1::ITensor* real_alpha_tensor = alpha_tensor;
    if (alpha_dims.nbDims != input_dims.nbDims) {
      auto* reshape_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *alpha_tensor);
      int c = alpha_dims.d[0];
      std::vector<nvinfer1::ITensor*> itensors;
      auto* n_tensor = Add1DConstantLayer(1);
      auto* c_tensor = Add1DConstantLayer(c);
      nvinfer1::ITensor* hw_tensor = nullptr;
      nvinfer1::ITensor* shape_tensor = nullptr;
      if (input_dims.nbDims - 2 > 0) {
        hw_tensor =
            Add1DConstantLayer(std::vector<int32_t>(input_dims.nbDims - 2, 1));
      }
      if (data_format == "NCHW") {
        if (hw_tensor != nullptr) {
          shape_tensor = Concat(
              std::vector<nvinfer1::ITensor*>{n_tensor, c_tensor, hw_tensor});
        } else {
          shape_tensor =
              Concat(std::vector<nvinfer1::ITensor*>{n_tensor, c_tensor});
        }
      } else {
        if (hw_tensor != nullptr) {
          shape_tensor = Concat(
              std::vector<nvinfer1::ITensor*>{n_tensor, hw_tensor, c_tensor});
        } else {
          shape_tensor =
              Concat(std::vector<nvinfer1::ITensor*>{n_tensor, c_tensor});
        }
      }
      reshape_layer->setInput(1, *shape_tensor);
      real_alpha_tensor = reshape_layer->getOutput(0);
    }

    nvinfer1::ILayer* layer = nullptr;

    layer = TRT_ENGINE_ADD_LAYER(
        engine_, ParametricReLU, *input, *real_alpha_tensor);

    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "prelu", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(prelu, PReluOpConverter);
