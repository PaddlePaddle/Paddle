// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>

#
#include "paddle/fluid/custom_engine/custom_device_load.h"
namespace paddle {

typedef bool (*RegisterDevicePluginFn)(CustomRuntimeParams* runtime_params);

typedef bool (*RegisterDevicePluginEngineFn)(CustomEngineParams* engine_params);

void LoadCustomLib(const std::string& dso_lib_path, void* dso_handle) {
  CustomRuntimeParams runtime_params;
  std::memset(&runtime_params, 0, sizeof(CustomRuntimeParams));
  runtime_params.size = sizeof(CustomRuntimeParams);
  auto device_interface = std::make_unique<C_DeviceInterface>();
  runtime_params.interface = device_interface.get();
  std::memset(runtime_params.interface, 0, sizeof(C_DeviceInterface));
  runtime_params.interface->size = sizeof(C_DeviceInterface);

  RegisterDevicePluginFn init_plugin_fn =
      reinterpret_cast<RegisterDevicePluginFn>(dlsym(dso_handle, "InitPlugin"));

  if (init_plugin_fn == nullptr) {
    LOG(WARNING) << "Skipped lib [" << dso_lib_path << "]: fail to find "
                 << "InitPlugin symbol in this lib.";
    return;
  }

  init_plugin_fn(&runtime_params);
  if (runtime_params.device_type == nullptr) {
    LOG(WARNING) << "Skipped lib [" << dso_lib_path
                 << "]: InitPlugin failed, please check the version "
                    "compatibility between PaddlePaddle and Custom Runtime.";
    return;
  }
  phi::LoadCustomRuntimeLib(
      runtime_params, std::move(device_interface), dso_lib_path, dso_handle);
  LOG(INFO) << "Succeed in loading custom runtime in lib: " << dso_lib_path;

  RegisterDevicePluginEngineFn init_plugin_engine_fn =
      reinterpret_cast<RegisterDevicePluginEngineFn>(
          dlsym(dso_handle, "InitPluginCustomEngine"));

  if (init_plugin_engine_fn == nullptr) {
    LOG(INFO) << "Skipped lib [" << dso_lib_path << "]: no custom engine "
              << "Plugin symbol in this lib.";
  } else {
    CustomEngineParams engine_params;
    std::memset(&engine_params, 0, sizeof(CustomEngineParams));
    engine_params.size = sizeof(CustomEngineParams);

    auto engine_interface = new (C_CustomEngineInterface);
    engine_params.interface = engine_interface;

    std::memset(engine_params.interface, 0, sizeof(C_CustomEngineInterface));
    engine_params.interface->size = sizeof(C_CustomEngineInterface);

    init_plugin_engine_fn(&engine_params);
    if (engine_params.interface == nullptr) {
      LOG(WARNING) << "Skipped lib [" << dso_lib_path
                   << "]: InitPluginEngine failed, please check the version "
                      "compatibility between PaddlePaddle and Custom Runtime.";
    } else {
      paddle::custom_engine::LoadCustomEngineLib(dso_lib_path, &engine_params);
      LOG(INFO) << "Succeed in loading custom engine in lib: " << dso_lib_path;
    }
  }
}
}  // namespace paddle
