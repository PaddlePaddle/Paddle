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
#include "paddle/fluid/inference/tensorrt/plugin/split_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class SplitOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert a fluid split op to tensorrt split layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto input_dims = input->getDimensions();
    size_t input_num = op_desc.Input("X").size();
    size_t output_num = op_desc.Output("Out").size();

    // Get Attrs
    int axis = BOOST_GET_CONST(int, op_desc.GetAttr("axis"));

    std::vector<int> output_lengths =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("sections"));
    int num = 0;
    if (op_desc.HasAttr("num")) {
      num = BOOST_GET_CONST(int, op_desc.GetAttr("num"));
    }

    if (engine_->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(6000)
      axis += (axis < 0) ? input_dims.nbDims : 0;
#endif
    } else {
      axis += (axis < 0) ? input_dims.nbDims : -1;
    }
    if (num > 0) {
      int64_t in_axis_dim = input_dims.d[axis];
      size_t out_axis_dim = in_axis_dim / num;
      for (int i = 0; i < num; ++i) {
        output_lengths.push_back(out_axis_dim);
      }
    }

    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::SplitPluginDynamic* plugin =
          new plugin::SplitPluginDynamic(axis, output_lengths, with_fp16);
      layer = engine_->AddDynamicPlugin(&input, input_num, plugin);
    } else {
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::SplitPlugin* plugin =
          new plugin::SplitPlugin(axis, output_lengths, with_fp16);
      layer = engine_->AddPluginV2Ext(&input, input_num, plugin);
    }

    std::string layer_name = "split (Output: ";
    for (size_t i = 0; i < output_num; i++) {
      auto output_name = op_desc.Output("Out")[i];
      layer->getOutput(i)->setName(output_name.c_str());
      engine_->SetITensor(output_name, layer->getOutput(i));
      layer_name += output_name;
      if (test_mode) {
        engine_->DeclareOutput(output_name);
      }
    }
    layer->setName((layer_name + ")").c_str());
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(split, SplitOpConverter);
