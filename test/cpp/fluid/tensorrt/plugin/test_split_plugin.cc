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

#include <gtest/gtest.h>

#include "paddle/fluid/inference/tensorrt/plugin/split_op_plugin.h"

namespace paddle::inference::tensorrt::plugin {

TEST(split_op_plugin, test_plugin) {
  int axis = 1;
  std::vector<int> output_lengths{1, 1};
  bool with_fp16 = false;
  std::vector<nvinfer1::DataType> input_types{nvinfer1::DataType::kFLOAT};
  std::vector<nvinfer1::Dims> input_dims;

  SplitPlugin sp_plugin(axis, output_lengths, with_fp16);
  nvinfer1::Dims in_dims;
  in_dims.nbDims = 4;
  input_dims.push_back(in_dims);
  sp_plugin.configurePlugin(input_dims.data(),
                            1,
                            nullptr,
                            2,
                            input_types.data(),
                            nullptr,
                            nullptr,
                            nullptr,
                            nvinfer1::PluginFormat::kLINEAR,
                            4);
  sp_plugin.initialize();
  sp_plugin.getPluginType();
  sp_plugin.canBroadcastInputAcrossBatch(0);
  sp_plugin.getNbOutputs();
  auto clone_plugin = sp_plugin.clone();
  clone_plugin->setPluginNamespace("test");
  clone_plugin->destroy();
  sp_plugin.getOutputDataType(0, input_types.data(), 1);
  sp_plugin.terminate();
}

TEST(split_op_plugin, test_plugin_creator) {
  SplitPluginCreator creator;
  creator.getFieldNames();
  creator.createPlugin("test", nullptr);
  creator.setPluginNamespace("test");
}

}  // namespace paddle::inference::tensorrt::plugin
