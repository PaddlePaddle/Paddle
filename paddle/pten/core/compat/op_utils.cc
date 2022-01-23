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

#include "paddle/pten/core/convert_utils.h"

#include "paddle/fluid/platform/enforce.h"

namespace pten {

DefaultKernelSignatureMap& DefaultKernelSignatureMap::Instance() {
  static DefaultKernelSignatureMap g_default_kernel_sig_map;
  return g_default_kernel_sig_map;
}

bool DefaultKernelSignatureMap::Has(const std::string& op_type) const {
  return map_.count(op_type) > 0;
}

const KernelSignature& DefaultKernelSignatureMap::Get(
    const std::string& op_type) const {
  auto it = map_.find(op_type);
  PADDLE_ENFORCE_NE(
      it,
      map_.end(),
      paddle::platform::errors::NotFound(
          "Operator `%s`'s kernel signature is not registered.", op_type));
  return it->second;
}

void DefaultKernelSignatureMap::Insert(std::string op_type,
                                       KernelSignature signature) {
  PADDLE_ENFORCE_NE(
      Has(op_type),
      true,
      paddle::platform::errors::AlreadyExists(
          "Operator (%s)'s Kernel Siginature has been registered.", op_type));
  map_.insert({std::move(op_type), std::move(signature)});
}

OpUtilsMap& OpUtilsMap::Instance() {
  static OpUtilsMap g_op_utils_map;
  return g_op_utils_map;
}

bool OpUtilsMap::Contains(const std::string& op_type) const {
  return name_map_.count(op_type) || arg_mapping_fn_map_.count(op_type);
}

void OpUtilsMap::InsertApiName(std::string op_type, std::string api_name) {
  PADDLE_ENFORCE_EQ(
      name_map_.count(op_type),
      0UL,
      paddle::platform::errors::AlreadyExists(
          "Operator (%s)'s api name has been registered.", op_type));
  name_map_.insert({std::move(op_type), std::move(api_name)});
}

void OpUtilsMap::InsertArgumentMappingFn(std::string op_type,
                                         ArgumentMappingFn fn) {
  PADDLE_ENFORCE_EQ(
      arg_mapping_fn_map_.count(op_type),
      0UL,
      paddle::platform::errors::AlreadyExists(
          "Operator (%s)'s argu,emt mapping function has been registered.",
          op_type));
  arg_mapping_fn_map_.insert({std::move(op_type), std::move(fn)});
}

std::string OpUtilsMap::GetApiName(const std::string& op_type) const {
  auto it = name_map_.find(op_type);
  if (it == name_map_.end()) {
    return "deprecated";
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
