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

#pragma once

#include <memory>
#include <unordered_map>

#include "NvInfer.h"
#include "paddle/fluid/inference/tensorrt/plugin/plugin_utils.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class PluginFactoryTensorRT : public nvinfer1::IPluginFactory {
 public:
  static PluginFactoryTensorRT* GetInstance() {
    static PluginFactoryTensorRT* factory_instance =
        new PluginFactoryTensorRT();
    return factory_instance;
  }

  // Deserialization method
  PluginTensorRT* createPlugin(const char* layer_name, const void* serial_data,
                               size_t serial_length) override;

  // Plugin construction, PluginFactoryTensorRT owns the plugin.
  PluginTensorRT* CreatePlugin(const std::string& op_name);

  bool RegisterPlugin(const std::string& op_name,
                      PluginDeserializeFunc deserialize_func,
                      PluginConstructFunc construct_func);

  bool IsPlugin(const std::string& op_name) {
    return plugin_registry_.find(op_name) != plugin_registry_.end();
  }

  size_t CountOwnedPlugins() { return owned_plugins_.size(); }

  void DestroyPlugins();

 protected:
  std::unordered_map<std::string,
                     std::pair<PluginDeserializeFunc, PluginConstructFunc>>
      plugin_registry_;
  std::vector<std::unique_ptr<PluginTensorRT>> owned_plugins_;
};

class TrtPluginRegistrar {
 public:
  TrtPluginRegistrar(const std::string& name,
                     PluginDeserializeFunc deserialize_func,
                     PluginConstructFunc construct_func) {
    auto factory = PluginFactoryTensorRT::GetInstance();
    // platform::PADDLE_ENFORCE(factory->RegisterPlugin(name, deserialize_func,
    // construct_func),  "Falied to register plugin [%s]", name);
    // platform::PADDLE_ENFORCE(factory->RegisterPlugin(name, deserialize_func,
    // construct_func));
    factory->RegisterPlugin(name, deserialize_func, construct_func);
  }
};

#define REGISTER_TRT_PLUGIN(name, deserialize_func, construct_func)    \
  REGISTER_TRT_PLUGIN_UNIQ_HELPER(__COUNTER__, name, deserialize_func, \
                                  construct_func)
#define REGISTER_TRT_PLUGIN_UNIQ_HELPER(ctr, name, deserialize_func, \
                                        construct_func)              \
  REGISTER_TRT_PLUGIN_UNIQ(ctr, name, deserialize_func, construct_func)
#define REGISTER_TRT_PLUGIN_UNIQ(ctr, name, deserialize_func, construct_func) \
  static ::paddle::inference::tensorrt::TrtPluginRegistrar                    \
      trt_plugin_registrar##ctr __attribute__((unused)) =                     \
          ::paddle::inference::tensorrt::TrtPluginRegistrar(                  \
              name, deserialize_func, construct_func)

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
