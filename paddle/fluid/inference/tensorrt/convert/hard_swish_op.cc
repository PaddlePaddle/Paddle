/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/inference/tensorrt/plugin/hard_swish_op_plugin.h"

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
 * HardSwish converter from fluid to tensorRT.
 */
class HardSwishOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert fluid HardSwish op to tensorrt HardSwish plugin";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    int input_num = op_desc.Input("X").size();
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    const float threshold =
        op_desc.HasAttr("threshold")
            ? PADDLE_GET_CONST(float, op_desc.GetAttr("threshold"))
            : 6.0f;
    const float scale = op_desc.HasAttr("scale")
                            ? PADDLE_GET_CONST(float, op_desc.GetAttr("scale"))
                            : 6.0f;
    const float offset =
        op_desc.HasAttr("offset")
            ? PADDLE_GET_CONST(float, op_desc.GetAttr("offset"))
            : 3.0f;
    nvinfer1::ILayer* layer = nullptr;
    if (threshold == scale) {
      auto* hsig_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Activation, *input, nvinfer1::ActivationType::kHARD_SIGMOID);
      hsig_layer->setAlpha(1.0 / scale);
      hsig_layer->setBeta(offset / scale);
      nvinfer1::IElementWiseLayer* eltwise_layer =
          TRT_ENGINE_ADD_LAYER(engine_,
                               ElementWise,
                               *input,
                               *(hsig_layer->getOutput(0)),
                               nvinfer1::ElementWiseOperation::kPROD);
      layer = eltwise_layer;
    } else {
      if (engine_->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(6000)
        plugin::HardSwishPluginDynamic* plugin =
            new plugin::HardSwishPluginDynamic(threshold, scale, offset);
        layer = engine_->AddDynamicPlugin(&input, input_num, plugin);
#else
        PADDLE_THROW(platform::errors::Fatal(
            "You are running the TRT Dynamic Shape mode, need to confirm that "
            "your TRT version is no less than 6.0"));
#endif
      } else {
        plugin::HardSwishPlugin* plugin =
            new plugin::HardSwishPlugin(threshold, scale, offset);
        layer = engine_->AddPlugin(&input, input_num, plugin);
      }
    }
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "hard_swish", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(hard_swish, HardSwishOpConverter);
