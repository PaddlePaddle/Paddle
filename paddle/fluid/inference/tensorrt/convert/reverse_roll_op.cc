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
#include "paddle/fluid/inference/tensorrt/plugin/reverse_roll_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
namespace paddle {
namespace inference {
namespace tensorrt {
class ReverseRollOpConverter : public OpConverter{
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a reverse_roll op to tensorrt "
               "reverse_roll plugin";
    framework::OpDesc op_desc(op, nullptr);

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    const int window_number = PADDLE_GET_CONST(int, op_desc.GetAttr("window_number"));
    const int window_size = PADDLE_GET_CONST(int, op_desc.GetAttr("window_size"));
    const int window_len = PADDLE_GET_CONST(int, op_desc.GetAttr("window_len"));
    const int shift_size = PADDLE_GET_CONST(int, op_desc.GetAttr("shift_size"));
    const int input_resolution = PADDLE_GET_CONST(int, op_desc.GetAttr("input_resolution"));
    PADDLE_ENFORCE_EQ(
        window_size*window_size,
        window_len,
        platform::errors::InvalidArgument(
            "The window_len should equal to window_size * window_size, but got window_size:%d, window_len:%d",
            window_size,window_len));
    PADDLE_ENFORCE_EQ(
        window_number*window_len,
        input_resolution*input_resolution,
        platform::errors::InvalidArgument(
            "The input_resolution*input_resolution should equal to window_number * window_len, but got window_len:%d, window_number:%d, input_resolution:%d",
            window_len,window_number,input_resolution));
    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
    nvinfer1::ILayer* reverse_roll_layer = nullptr;
    if (engine_->with_dynamic_shape()) {
        plugin::ReverseRollPluginDynamic* plugin =
            new plugin::ReverseRollPluginDynamic(window_number,
                window_len,
                window_size,
                input_resolution,
                shift_size,
                with_fp16);
        reverse_roll_layer = engine_->AddDynamicPlugin(&X, 1, plugin);
    } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
          "ReverseROll TRT Plugin should run in dynamic shape."));
    }
    auto output_name = op_desc.Output("Out").front();
    RreplenishLayerAndOutput(
        reverse_roll_layer, "reverse_roll", {output_name}, test_mode);

  } 
};

}
}
}

REGISTER_TRT_OP_CONVERTER(reverse_roll,
                          ReverseRollOpConverter);
