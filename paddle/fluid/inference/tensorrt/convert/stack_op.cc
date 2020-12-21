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
#include "paddle/fluid/inference/tensorrt/plugin/stack_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * Stack converter from fluid to tensorRT.
 */
class StackOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert fluid stack op to tensorrt stack layer";

    framework::OpDesc op_desc(op, nullptr);
    auto input = op_desc.Input("X");
    int input_num = input.size();
    nvinfer1::ITensor** inputs =
        (nvinfer1::ITensor**)malloc(input_num * sizeof(nvinfer1::ITensor*));

    for (int i = 0; i < input_num; ++i) {
      inputs[i] = engine_->GetITensor(input[i]);
    }

    int axis = BOOST_GET_CONST(int, op_desc.GetAttr("axis"));
    if (axis < 0) {
      axis = axis + inputs[0]->getDimensions().nbDims + 1;
    }

    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(6000)
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::StackPluginDynamic* plugin =
          new plugin::StackPluginDynamic(axis, input_num, with_fp16);
      layer = engine_->AddPluginV2(inputs, input_num, plugin);
      assert(layer != nullptr);
#else
      PADDLE_THROW(platform::errors::Fatal(
          "You are running the TRT Dynamic Shape mode, need to confirm that "
          "your TRT version is no less than 6.0"));
#endif
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "You are running the Ernie(Bert) model in static"
          "shape mode, which is not supported for the time being.\n"
          "You can use the config.SetTRTDynamicShapeInfo(...) interface"
          " to set the shape information to run the dynamic shape mode."));
    }
    auto output_name = op_desc.Output("Y").front();
    RreplenishLayerAndOutput(layer, "stack", {output_name}, test_mode);
    free(inputs);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(stack, StackOpConverter);
