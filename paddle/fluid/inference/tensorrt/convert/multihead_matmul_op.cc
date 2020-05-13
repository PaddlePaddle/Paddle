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
#include "paddle/fluid/inference/tensorrt/plugin/qkv_to_context_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class MultiheadMatMulOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
#if IS_TRT_VERSION_GE(6000)
    VLOG(3) << "convert a fluid multihead_mamul op to a corresponding tensorrt "
               "network structure";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    // Shouble be a 5 dims tensor.
    auto* input = engine_->GetITensor(op_desc.Input("Input").front());
    auto* input_bias_qk = engine_->GetITensor(op_desc.Input("BiasQK").front());

    // fc weights and fc bias
    auto weight_name = op_desc.Input("W").front();
    auto bias_name = op_desc.Input("Bias").front();

    auto* weight_v = scope.FindVar(weight_name);
    auto* weight_t = weight_v->GetMutable<framework::LoDTensor>();

    auto* bias_v = scope.FindVar(bias_name);
    auto* bias_t = bias_v->GetMutable<framework::LoDTensor>();

    float* weight_data =
        engine_->GetWeightCPUData(weight_name, weight_t, false);
    float* bias_data = engine_->GetWeightCPUData(bias_name, bias_t, false);
    std::vector<float> weight_data_tmp;
    weight_data_tmp.reserve(weight_t->numel());
    memcpy(weight_data_tmp.data(), weight_data,
           weight_t->numel() * sizeof(float));

    //  (hidden, 3, all_head_size)
    auto weight_dims = weight_t->dims();

    int hidden = weight_dims[0];         // channels_in
    int three = weight_dims[1];          // channels_out
    int all_head_size = weight_dims[2];  // channels_out
    int m = hidden;
    int n = three * all_head_size;
    auto tranpose_weight = [](const float* src, float* dst, int m, int n) {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          dst[j * m + i] = src[i * n + j];
        }
      }
    };

    // transpose weight_data from m * n to  n * m
    tranpose_weight(weight_data_tmp.data(), weight_data, m, n);
    TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(weight_data),
                                  static_cast<size_t>(weight_t->numel())};

    weight.dims.assign({n, m});
    TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT,
                                static_cast<void*>(bias_data),
                                static_cast<size_t>(bias_t->numel())};

    auto* fc_layer = TRT_ENGINE_ADD_LAYER(engine_, FullyConnected, *input, n,
                                          weight.get(), bias.get());
    auto* fc_out = fc_layer->getOutput(0);
    // add qkv to context
    int head_number = BOOST_GET_CONST(int, op_desc.GetAttr("head_number"));
    int head_size = all_head_size / head_number;
    float scale = BOOST_GET_CONST(float, op_desc.GetAttr("alpha"));

    std::vector<nvinfer1::ITensor*> plugin_inputs;
    plugin_inputs.push_back(fc_out);
    plugin_inputs.push_back(input_bias_qk);
    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      bool ban_fp16 = engine_->disable_trt_plugin_fp16();
      plugin::DynamicPluginTensorRT* plugin =
          new plugin::QkvToContextPluginDynamic(hidden, head_number, head_size,
                                                scale, ban_fp16);
      layer = engine_->AddPluginV2(plugin_inputs.data(), 2, plugin);
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "You are running the Ernie(Bert) model in static shape mode, which "
          "is not supported for the time being.\n"
          "You can use the config.SetTRTDynamicShapeInfo(...) interface to set "
          "the shape information to run the dynamic shape mode."));
    }
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "multihead_matmul", {output_name},
                             test_mode);
#else
    PADDLE_THROW(platform::errors::Fatal(
        "You are running the TRT Dynamic Shape mode, need to confirm that "
        "your TRT version is no less than 6.0"));
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(multihead_matmul, MultiheadMatMulOpConverter);
