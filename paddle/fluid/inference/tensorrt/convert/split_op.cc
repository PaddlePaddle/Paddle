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
    PADDLE_ENFORCE_EQ(input_num, 1UL,
                      platform::errors::InvalidArgument(
                          "Invalid input X's size of split TRT converter. "
                          "Expected 1, received %d.",
                          input_num));
    int axis = BOOST_GET_CONST(int, op_desc.GetAttr("axis"));
    // split on batch is not supported in TensorRT
    PADDLE_ENFORCE_NE(
        axis, 0,
        platform::errors::InvalidArgument(
            "Invalid split axis. Split on batch is not supported in TensorRT"));

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

    PADDLE_ENFORCE_NE(input_dims.d[axis], -1,
                      platform::errors::InvalidArgument(
                          "The (%d) dim of input should not be -1", axis));
    if (num > 0) {
      int64_t in_axis_dim = input_dims.d[axis];
      PADDLE_ENFORCE_EQ(
          in_axis_dim % num, 0,
          platform::errors::InvalidArgument(
              "Invalid number to split. Tensor split does not result"
              " in an equal division of dimensions. Axis dim = %d %% num = %d "
              "!= 0",
              in_axis_dim, num));
      size_t out_axis_dim = in_axis_dim / num;
      for (int i = 0; i < num; ++i) {
        output_lengths.push_back(out_axis_dim);
      }
    }

    PADDLE_ENFORCE_EQ(
        output_lengths.size(), output_num,
        platform::errors::InvalidArgument(
            "The output_length should be equal to the output size."));

    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(6000)
      plugin::SplitPluginDynamic* plugin =
          new plugin::SplitPluginDynamic(axis, output_lengths);
      layer = engine_->AddPluginV2(&input, input_num, plugin);
#else
      PADDLE_THROW(platform::errors::Fatal(
          "You are running the TRT Dynamic Shape mode, need to confirm that "
          "your TRT version is no less than 6.0"));
#endif
    } else {
      plugin::SplitPlugin* plugin =
          new plugin::SplitPlugin(axis, output_lengths);
      layer = engine_->AddPlugin(&input, input_num, plugin);
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
