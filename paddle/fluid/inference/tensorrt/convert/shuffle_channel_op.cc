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
 * ConcatOp
 */
class ShuffleChannelOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto input_dims = input->getDimensions();
    auto output_name = op_desc.Output("Out")[0];
    int group = BOOST_GET_CONST(int, op_desc.GetAttr("group"));

    if (engine_->with_dynamic_shape()) {
      auto* input_shape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shape, *input);
      auto* input_shape_tensor = input_shape_layer->getOutput(0);
      auto* channel_index_tensor =
          Add1DConstantLayer(1, output_name + "_channel_index_tensor_");
      auto* channel_shape_tensor =
          TRT_ENGINE_ADD_LAYER(engine_, Gather, *input_shape_tensor,
                               *channel_index_tensor, 0)
              ->getOutput(0);
      auto* group_tensor =
          Add1DConstantLayer(group, output_name + "_group_tensor_");

      auto* new_channel_shape_tensor =
          TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *channel_shape_tensor,
                               *group_tensor,
                               nvinfer1::ElementWiseOperation::kDIV)
              ->getOutput(0);

      std::vector<int32_t> shape_dim3{0, 2, 3};
      auto* shape_dim3_index_tensor =
          Add1DConstantLayer(shape_dim3, output_name + "_shape_dim3_tensor_");
      auto* shape_dim3_tensor =
          TRT_ENGINE_ADD_LAYER(engine_, Gather, *input_shape_tensor,
                               *shape_dim3_index_tensor, 0)
              ->getOutput(0);

      std::vector<nvinfer1::ITensor*> itensors;
      itensors.push_back(shape_dim3_tensor);
      itensors.push_back(group_tensor);
      itensors.push_back(new_channel_shape_tensor);
      auto* reshape_tensor =
          TRT_ENGINE_ADD_LAYER(engine_, Concatenation, itensors.data(),
                               itensors.size())
              ->getOutput(0);

      auto* reshape_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *reshape_tensor);
      nvinfer1::Permutation transpose_new_input{0, 3, 4, 1, 2};
      reshape_layer->setSecondTranspose(transpose_new_input);

      auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      layer->setInput(1, *(reshape_layer->getOutput(0)));
      nvinfer1::Permutation transpose_embed{0, 2, 1, 3, 4};
      layer->setSecondTranspose(transpose_embed);
      auto* output = layer->getOutput(0);
      auto* output_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *output);
      output_layer->setInput(1, *input_shape_tensor);

      RreplenishLayerAndOutput(output_layer, "shuffle_channel", {output_name},
                               test_mode);
    } else {
      int c = input_dims.d[0];
      int h = input_dims.d[1];
      int w = input_dims.d[2];

      auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      nvinfer1::Dims4 reshape_dim(group, c / group, h, w);
      layer->setReshapeDimensions(reshape_dim);
      layer->setSecondTranspose({1, 0, 2, 3});
      auto* output = layer->getOutput(0);

      auto* reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *output);
      nvinfer1::Dims3 reshape_dim2(c, h, w);
      reshape_layer->setReshapeDimensions(reshape_dim2);

      RreplenishLayerAndOutput(reshape_layer, "shuffle_channel", {output_name},
                               test_mode);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(shuffle_channel, ShuffleChannelOpConverter);
