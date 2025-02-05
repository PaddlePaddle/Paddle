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
#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/backends/device_manager.h"

#include "paddle/fluid/custom_engine/custom_engine_ext.h"
#include "paddle/fluid/custom_engine/custom_engine_manager.h"
namespace paddle {

void LoadCustomLib(const std::string& dso_lib_path, void* dso_handle);
}  // namespace paddle
