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

#pragma once

#include "paddle/pten/api/ext/op_kernel_info.h"

namespace paddle {
namespace framework {

// Load custom kernel lib from giwen path
void LoadCustomKernel(const std::string& libs_path);

void LoadCustomKernelLib(const std::string& dso_lib_path);

// Load custom kernel api: register kernel after user compiled
void LoadOpKernelInfoAndRegister(const std::string& dso_name);

// Register custom kernel api: register kernel directly
void RegisterKernelWithMetaInfoMap(
    const paddle::OpKernelInfoMap& op_kernel_info_map);

// Interface for selective register custom kernel.
void RegisterKernelWithMetaInfo(
    const std::vector<OpKernelInfo>& op_kernel_infos);
}  // namespace framework
}  // namespace paddle
