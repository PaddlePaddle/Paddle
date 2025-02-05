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

#include "paddle/fluid/custom_engine/custom_engine_manager.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/pir/include/core/ir_context.h"

namespace paddle {
namespace custom_engine {
bool ValidCustomCustomEngineParams(const CustomEngineParams* params) {
#define CHECK_INTERFACE(ptr, required)                                  \
  if (params->interface->ptr == nullptr && required) {                  \
    LOG(WARNING) << "CustomEngine pointer: " << #ptr << " is not set."; \
    return false;                                                       \
  }

  CHECK_INTERFACE(graph_engine_build, true);
  CHECK_INTERFACE(graph_engine_execute, true);
  CHECK_INTERFACE(custom_engine_op_lower, true);

  return true;
#undef CHECK_INTERFACE
}

void LoadCustomEngineLib(const std::string& dso_lib_path,
                         CustomEngineParams* engine_params) {
  if (ValidCustomCustomEngineParams(engine_params)) {
    paddle::custom_engine::CustomEngineManager::Instance()
        ->SetCustomEngineInterface(engine_params->interface);

    auto* interface = paddle::custom_engine::CustomEngineManager::Instance()
                          ->GetCustomEngineInterface();

    // register custom engine op
    if (interface->register_custom_engine_op) {
      interface->register_custom_engine_op();
    } else {
      LOG(WARNING) << "Skipped lib [" << dso_lib_path
                   << "]. register_custom_engine_op is not set!!! please check "
                      "the version "
                      "compatibility between PaddlePaddle and Custom Engine.";
    }

  } else {
    LOG(WARNING) << "Skipped lib [" << dso_lib_path
                 << "]. Wrong engine parameters!!! please check the version "
                    "compatibility between PaddlePaddle and Custom Engine.";
  }
}

}  // namespace custom_engine
}  // namespace paddle
