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

#pragma once

#include <NvInfer.h>
#include <cstring>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_utils.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class PluginFactoryTensorRT : public nvinfer1::IPluginFactory,
                              public DeleteHelper {
 public:
  // Deserialization method
  PluginTensorRT* createPlugin(const char* layer_name, const void* serial_data,
                               size_t serial_length) override;

  bool RegisterPlugin(const std::string& op_name,
                      PluginDeserializeFunc deserialize_func);

  bool Has(const std::string& op_name) {
    return plugin_registry_.find(op_name) != plugin_registry_.end();
  }

  void DestroyPlugins();

 protected:
  std::unordered_map<std::string, PluginDeserializeFunc> plugin_registry_;

  std::list<std::unique_ptr<PluginTensorRT>> owned_plugins_;
};

class TrtPluginRegistrar {
 public:
  TrtPluginRegistrar(const std::string& name,
                     PluginDeserializeFunc deserialize_func) {
    inference::Singleton<PluginFactoryTensorRT>::Global().RegisterPlugin(
        name, deserialize_func);
  }
};

#define REGISTER_TRT_PLUGIN(name, deserialize_func) \
  REGISTER_TRT_PLUGIN_UNIQ(__COUNTER__, name, deserialize_func)

#define REGISTER_TRT_PLUGIN_UNIQ(ctr, name, deserialize_func)      \
  static paddle::inference::tensorrt::plugin::TrtPluginRegistrar   \
      trt_plugin_registrar##ctr UNUSED =                           \
          paddle::inference::tensorrt::plugin::TrtPluginRegistrar( \
              name, deserialize_func)

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
