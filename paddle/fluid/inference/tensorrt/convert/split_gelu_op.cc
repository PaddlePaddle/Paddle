/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/inference/tensorrt/plugin/split_gelu_op_plugin.h"
namespace paddle {
namespace inference {
namespace tensorrt {
class SplitGeluOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a split_gelu op to tensorrt "
               "split_gelu plugin";
    framework::OpDesc op_desc(op, nullptr);

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
    nvinfer1::ILayer* split_gelu_layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      plugin::SplitGeluPluginDynamic* plugin =
          new plugin::SplitGeluPluginDynamic(with_fp16);
      split_gelu_layer = engine_->AddDynamicPlugin(&X, 1, plugin);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "ReverseROll TRT Plugin should run in dynamic shape."));
    }
    auto output_name = op_desc.Output("Out").front();
    RreplenishLayerAndOutput(
        split_gelu_layer, "split_gelu", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(split_gelu, SplitGeluOpConverter);
