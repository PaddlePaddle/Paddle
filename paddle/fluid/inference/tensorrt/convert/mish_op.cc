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

namespace paddle::inference::tensorrt {

/*
 * Mish OP
 */
class MishOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert mish op to tensorrt mish plugin";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    int input_num = op_desc.Input("X").size();
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    const float threshold =
        op_desc.HasAttr("threshold")
            ? PADDLE_GET_CONST(float, op_desc.GetAttr("threshold"))
            : 20.0f;

    nvinfer1::ILayer* layer = nullptr;
    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
    plugin::MishPluginDynamic* plugin =
        new plugin::MishPluginDynamic(threshold, with_fp16);
    layer = engine_->AddDynamicPlugin(&input, input_num, plugin);

    auto output_name = op_desc.Output("Out")[0];
    ReplenishLayerAndOutput(layer, "mish", {output_name}, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(mish, MishOpConverter);
