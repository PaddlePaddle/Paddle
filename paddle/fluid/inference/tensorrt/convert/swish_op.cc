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
#include "paddle/fluid/inference/tensorrt/plugin/swish_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class SwishOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert fluid swish op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    int input_num = op_desc.Input("X").size();
    PADDLE_ENFORCE(input_num == 1);
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    // Get output
    size_t output_num = op_desc.Output("Out").size();
    PADDLE_ENFORCE(output_num == 1);
    // Get attrs
    float beta = BOOST_GET_CONST(float, op_desc.GetAttr("beta"));

    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(6000)
      plugin::SwishPluginDynamic* plugin = new plugin::SwishPluginDynamic(beta);
      layer = engine_->AddPluginV2(&input, input_num, plugin);
#else
      PADDLE_THROW(platform::errors::Fatal(
          "You are running the TRT Dynamic Shape mode, need to confirm that "
          "your TRT version is no less than 6.0"));
#endif
    } else {
      plugin::SwishPlugin* plugin = new plugin::SwishPlugin(beta);
      layer = engine_->AddPlugin(&input, input_num, plugin);
    }

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "swish", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(swish, SwishOpConverter);
