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
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert fluid prelu op to tensorrt prelu layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    size_t input_num = op_desc.Input("X").size();
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    // Get attrs
    std::string mode = BOOST_GET_CONST(std::string, op_desc.GetAttr("mode"));
    std::string data_format = "NCHW";
    if (op_desc.HasAttr("data_format")) {
      data_format =
          BOOST_GET_CONST(std::string, op_desc.GetAttr("data_format"));
    }
    bool trt_nhwc_convert =
        op_desc.HasAttr("trt_nhwc_convert") &&
        BOOST_GET_CONST(bool, op_desc.GetAttr("trt_nhwc_convert"));
    auto* alpha_var = scope.FindVar(op_desc.Input("Alpha")[0]);
    auto* alpha_tensor = alpha_var->GetMutable<framework::LoDTensor>();
    auto alpha_tensor_dim = alpha_tensor->dims();
    platform::CPUPlace cpu_place;
    // Alpha has 1 (all) or 4 (channel, element) dimensions.
    // We only transpose alpha when the shape of alpha is 4.
    if (trt_nhwc_convert && alpha_tensor_dim.size() == 4) {
      // Convert alpha from NHWC to NCHW
      VLOG(4) << "Convert alpha from NHWC to NCHW";
      float* alpha_data = alpha_tensor->mutable_data<float>(cpu_place);
      int N = alpha_tensor_dim[0];
      int H = alpha_tensor_dim[1];
      int W = alpha_tensor_dim[2];
      int C = alpha_tensor_dim[3];
      std::vector<float> alpha_data_copy(alpha_data,
                                         alpha_data + alpha_tensor->numel());
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < C; ++j) {
          for (int k = 0; k < H; ++k) {
            for (int l = 0; l < W; ++l) {
              alpha_data[i * C * H * W + j * H * W + k * W + l] =
                  alpha_data_copy[i * H * W * C + k * W * C + l * C + j];
            }
          }
        }
      }
      framework::DDim nchw_dim(alpha_tensor->dims());
      nchw_dim[1] = C;
      nchw_dim[2] = H;
      nchw_dim[3] = W;
      alpha_tensor->Resize(nchw_dim);
    }

    std::unique_ptr<framework::LoDTensor> alpha_tensor_temp(
        new framework::LoDTensor());
    alpha_tensor_temp->Resize(alpha_tensor->dims());
    paddle::framework::TensorCopySync(*alpha_tensor, cpu_place,
                                      alpha_tensor_temp.get());
    float* alpha_data = alpha_tensor_temp->mutable_data<float>(cpu_place);

    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      plugin::PReluPluginDynamic* plugin = new plugin::PReluPluginDynamic(
          alpha_data, alpha_tensor_temp->numel(), mode, data_format);
      layer = engine_->AddDynamicPlugin(&input, input_num, plugin);
    } else {
#if IS_TRT_VERSION_GE(7000)
      float* alpha_weight_data =
          engine_->GetWeightCPUData(op_desc.Input("Alpha")[0], alpha_tensor);
      TensorRTEngine::Weight alpha_weight{
          nvinfer1::DataType::kFLOAT, static_cast<void*>(alpha_weight_data),
          static_cast<size_t>(alpha_tensor->numel())};

      nvinfer1::Dims dims;
      dims.nbDims = 0;
      // jump batch dim
      for (int i = 1; i < alpha_tensor->dims().size(); i++) {
        dims.d[dims.nbDims++] = alpha_tensor->dims()[i];
      }
      for (; dims.nbDims < input->getDimensions().nbDims; dims.nbDims++) {
        dims.d[dims.nbDims] = 1;
      }

      auto alpha_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Constant, dims, alpha_weight.get());
      auto alpha_layer_output = alpha_layer->getOutput(0);

      layer = TRT_ENGINE_ADD_LAYER(engine_, ParametricReLU, *input,
                                   *alpha_layer_output);
#else
      plugin::PReluPlugin* plugin = new plugin::PReluPlugin(
          alpha_data, alpha_tensor_temp->numel(), mode, data_format);
      layer = engine_->AddPlugin(&input, input_num, plugin);
#endif
    }
    // keep alpha tensor to avoid release it's memory
    engine_->SetWeights(op_desc.Input("Alpha")[0],
                        std::move(alpha_tensor_temp));

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "prelu", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(prelu, PReluOpConverter);
