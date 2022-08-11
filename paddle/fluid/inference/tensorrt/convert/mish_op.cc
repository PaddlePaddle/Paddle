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
#include "paddle/fluid/inference/tensorrt/plugin/mish_op_plugin.h"

namespace paddle {
namespace framework {
class Scope;

namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * Mish converter from fluid to tensorRT.
 */
class MishOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert fluid Mish op to tensorrt Mish plugin";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    const float threshold =
        op_desc.HasAttr("threshold")
            ? PADDLE_GET_CONST(float, op_desc.GetAttr("threshold"))
            : 20.0f;
    nvinfer1::ILayer* layer = nullptr;

#if IS_TRT_VERSION_GE(5130)
    // mish -> clip + softplus + tanh + prod
    auto* layer_clip = TRT_ENGINE_ADD_LAYER(
        engine_, Activation, *input, nvinfer1::ActivationType::kCLIP);
    layer_clip->setAlpha(-3.40282e+038);
    layer_clip->setBeta(threshold);
    auto* softplus_layer =
        TRT_ENGINE_ADD_LAYER(engine_,
                             Activation,
                             *layer_clip->getOutput(0),
                             nvinfer1::ActivationType::kSOFTPLUS);
    softplus_layer->setAlpha(1.0f);
    softplus_layer->setBeta(1.0f);
    auto* tan_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                           Activation,
                                           *softplus_layer->getOutput(0),
                                           nvinfer1::ActivationType::kTANH);
    layer = TRT_ENGINE_ADD_LAYER(engine_,
                                 ElementWise,
                                 *input,
                                 *(tan_layer->getOutput(0)),
                                 nvinfer1::ElementWiseOperation::kPROD);
#else
    int input_num = op_desc.Input("X").size();
    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
    plugin::MishPlugin* plugin = new plugin::MishPlugin(threshold, with_fp16);
    layer = engine_->AddPlugin(&input, input_num, plugin);
#endif

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "mish", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(mish, MishOpConverter);
