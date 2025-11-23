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

#include <gtest/gtest.h>

#include "paddle/fluid/inference/tensorrt/plugin/fused_token_prune_op_plugin.h"

namespace paddle::inference::tensorrt::plugin {

TEST(fused_token_prune_op_plugin, test_plugin) {
  FusedTokenPrunePluginDynamic plugin(true,
                                      /*keep_first_token*/ false,
                                      /*keep_order*/ true,
                                      /*flag_varseqlen*/ false);
  plugin.initialize();
  plugin.getPluginType();
  plugin.getNbOutputs();
  size_t buf_size = plugin.getSerializationSize();
  std::vector<char> buf(buf_size);
  plugin.serialize(buf.data());
}

TEST(fused_token_prune_op_plugin, test_plugin_creator) {
  FusedTokenPrunePluginDynamicCreator creator;
  creator.getFieldNames();
  creator.createPlugin("test", nullptr);
  creator.setPluginNamespace("test");
}

}  // namespace paddle::inference::tensorrt::plugin
