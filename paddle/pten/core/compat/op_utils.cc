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

#include "paddle/pten/core/compat/op_utils.h"

namespace pten {

DefaultKernelSignatureMap& DefaultKernelSignatureMap::Instance() {
  static DefaultKernelSignatureMap g_default_kernel_sig_map;
  return g_default_kernel_sig_map;
}

OpUtilsMap& OpUtilsMap::Instance() {
  static OpUtilsMap g_op_utils_map;
  return g_op_utils_map;
}

std::string OpUtilsMap::GetBaseKernelName(const std::string& op_type) const {
  if (deprecated_op_names.find(op_type) != deprecated_op_names.end()) {
    return "deprecated";
  }
  auto it = name_map_.find(op_type);
  if (it == name_map_.end()) {
    return op_type;
  } else {
    return it->second;
  }
}

ArgumentMappingFn OpUtilsMap::GetArgumentMappingFn(
    const std::string& op_type) const {
  auto it = arg_mapping_fn_map_.find(op_type);
  if (it == arg_mapping_fn_map_.end()) {
    auto func =
        [op_type](const ArgumentMappingContext& ctx) -> KernelSignature {
      return DefaultKernelSignatureMap::Instance().Get(op_type);
    };
    return func;
  } else {
    return it->second;
  }
}

}  // namespace pten
