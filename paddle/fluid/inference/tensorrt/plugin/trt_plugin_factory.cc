// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

PluginTensorRT* PluginFactoryTensorRT::createPlugin(const char* layer_name,
                                                    const void* serial_data,
                                                    size_t serial_length) {
  const char* plugin_type;
  DeserializeValue(&serial_data, &serial_length, &plugin_type);

  PADDLE_ENFORCE(Has(plugin_type),
                 "trt plugin type %s does not exists, check it.", plugin_type);
  auto plugin = plugin_registry_[plugin_type](serial_data, serial_length);
  owned_plugins_.emplace_back(plugin);

  return plugin;
}

bool PluginFactoryTensorRT::RegisterPlugin(
    const std::string& op_name, PluginDeserializeFunc deserialize_func) {
  if (Has(op_name)) return false;
  auto ret = plugin_registry_.emplace(op_name, deserialize_func);
  return ret.second;
}

void PluginFactoryTensorRT::DestroyPlugins() { owned_plugins_.clear(); }

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
