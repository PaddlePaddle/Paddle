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

#include "paddle/fluid/inference/tensorrt/plugin/plugin_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {

PluginTensorRT* PluginFactoryTensorRT::createPlugin(const char* layer_name,
                                                    const void* serial_data,
                                                    size_t serial_length) {
  size_t parsed_byte = 0;
  std::string encoded_op_name =
      ExtractOpName(serial_data, serial_length, &parsed_byte);

  if (!IsPlugin(encoded_op_name)) {
    return nullptr;
  }

  auto plugin_ptr =
      plugin_registry_[encoded_op_name].first(serial_data, serial_length);
  owned_plugins_.emplace_back(plugin_ptr);

  return plugin_ptr;
}

PluginTensorRT* PluginFactoryTensorRT::CreatePlugin(
    const std::string& op_name) {
  if (!IsPlugin(op_name)) return nullptr;

  auto plugin_ptr = plugin_registry_[op_name].second();
  owned_plugins_.emplace_back(plugin_ptr);

  return plugin_ptr;
}

bool PluginFactoryTensorRT::RegisterPlugin(
    const std::string& op_name, PluginDeserializeFunc deserialize_func,
    PluginConstructFunc construct_func) {
  if (IsPlugin(op_name)) return false;

  auto ret = plugin_registry_.emplace(
      op_name, std::make_pair(deserialize_func, construct_func));

  return ret.second;
}

void PluginFactoryTensorRT::DestroyPlugins() { owned_plugins_.clear(); }

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
