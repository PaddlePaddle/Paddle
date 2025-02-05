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

#pragma once
#include <cstdint>
#include <string>

#include "paddle/fluid/custom_engine/custom_engine_ext.h"

namespace paddle {
namespace custom_engine {

class CustomEngineManager {
 public:
  static CustomEngineManager* Instance();

  CustomEngineManager();
  ~CustomEngineManager();

  C_CustomEngineInterface* GetCustomEngineInterface();

  void SetCustomEngineInterface(
      C_CustomEngineInterface* custom_engine_interface);

 private:
  C_CustomEngineInterface* interface_;
};

void LoadCustomEngineLib(const std::string& dso_lib_path,
                         CustomEngineParams* engine_params);

}  // namespace custom_engine
}  // namespace paddle
