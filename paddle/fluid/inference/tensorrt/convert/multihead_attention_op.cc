/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See
the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/qkv_to_context_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class MultiheadAttentionOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3)
        << "convert a fluid multihead_attention op to a corresponding tensorrt "
           "network structure";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input_q = engine_->GetITensor(op_desc.Input("Q")[0]);  // B, S, N, H
    auto* input_k = engine_->GetITensor(op_desc.Input("K")[0]);
    auto* input_v = engine_->GetITensor(op_desc.Input("V")[0]);

    VLOG(3) << "input q:";
    auto dims = input_q->getDimensions();
    VLOG(3) << "nbDims: " << dims.nbDims << "; dims: " << dims.d[0] << ","
            << dims.d[1] << "," << dims.d[2] << "," << dims.d[3] << ","
            << dims.d[4];

    VLOG(3) << "input k:";
    dims = input_k->getDimensions();
    VLOG(3) << "nbDims: " << dims.nbDims << "; dims: " << dims.d[0] << ","
            << dims.d[1] << "," << dims.d[2] << "," << dims.d[3] << ","
            << dims.d[4];

    VLOG(3) << "input v:";
    dims = input_v->getDimensions();
    VLOG(3) << "nbDims: " << dims.nbDims << "; dims: " << dims.d[0] << ","
            << dims.d[1] << "," << dims.d[2] << "," << dims.d[3] << ","
            << dims.d[4];

    VLOG(3) << "step1";
    bool enable_int8 = op_desc.HasAttr("enable_int8");
    int head_number = BOOST_GET_CONST(int, op_desc.GetAttr("head_number"));

    nvinfer1::ILayer* layer = nullptr;
    auto output_name = op_desc.Output("Out")[0];

    if (engine_->with_dynamic_shape()) {
      if (engine_->use_oss()) {
      } else {
        // Declare inputs
        std::vector<nvinfer1::ITensor*> itensors(
            {input_q, input_k, input_v});  // B, S, N, H
        VLOG(3) << "step2";
        auto* concat_layer =
            TRT_ENGINE_ADD_LAYER(engine_, Concatenation, itensors.data(),
                                 itensors.size());  // B, S, 3*N, H
        concat_layer->setAxis(2);

        VLOG(3) << "concat output:";
        dims = concat_layer->getOutput(0)->getDimensions();
        VLOG(3) << "nbDims: " << dims.nbDims << "; dims: " << dims.d[0] << ","
                << dims.d[1] << "," << dims.d[2] << "," << dims.d[3] << ","
                << dims.d[4];

        VLOG(3) << "step3";
        auto* shuffle_layer =
            TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *concat_layer->getOutput(0));
        VLOG(3) << "step4";
        nvinfer1::Dims shape_dim;
        shape_dim.nbDims = 5;
        shape_dim.d[0] = 0;
        shape_dim.d[1] = 0;
        shape_dim.d[2] = 0;
        shape_dim.d[3] = 1;
        shape_dim.d[4] = 1;
        int head_size = input_q->getDimensions().d[2] / head_number;
        shuffle_layer->setReshapeDimensions(shape_dim);

        VLOG(3) << "reshape output:";
        dims = shuffle_layer->getOutput(0)->getDimensions();
        VLOG(3) << "nbDims: " << dims.nbDims << "; dims: " << dims.d[0] << ","
                << dims.d[1] << "," << dims.d[2] << "," << dims.d[3] << ","
                << dims.d[4];

        VLOG(3) << "step5";
        auto* input_bias_qk =
            engine_->GetITensor(op_desc.Input("BiasQK").front());

        // add qkv to context
        float scale = BOOST_GET_CONST(float, op_desc.GetAttr("alpha"));

        std::vector<nvinfer1::ITensor*> plugin_inputs;
        plugin_inputs.push_back(shuffle_layer->getOutput(0));
        plugin_inputs.push_back(input_bias_qk);
        bool with_fp16 =
            engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();

        if (enable_int8) {
          with_fp16 = 1;
        }
        VLOG(3) << "head_number: " << head_number
                << "; head_size: " << head_size;
        plugin::DynamicPluginTensorRT* plugin =
            new plugin::QkvToContextPluginDynamic(head_number * head_size,
                                                  head_number, head_size, scale,
                                                  with_fp16);
        layer = engine_->AddDynamicPlugin(plugin_inputs.data(), 2, plugin);
      }
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "You are running the Ernie(Bert) model in static shape mode, which "
          "is not supported for the time being.\n"
          "You can use the config.SetTRTDynamicShapeInfo(...) interface to set "
          "the shape information to run the dynamic shape mode."));
    }
    RreplenishLayerAndOutput(layer, "multihead_attention", {output_name},
                             test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(multihead_attention, MultiheadAttentionOpConverter);
