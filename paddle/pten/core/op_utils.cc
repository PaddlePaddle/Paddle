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

#include "paddle/pten/core/op_utils.h"

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
  return op_utils_map_.count(op_type) > 0;
}

const OpUtils& OpUtilsMap::Get(const std::string& op_type) const {
  PADDLE_ENFORCE_EQ(
      Contains(op_type),
      true,
      paddle::platform::errors::NotFound(
          "Operator (%s)'s compatible utils is not registered.", op_type));
  return op_utils_map_.at(op_type);
}

OpUtils* OpUtilsMap::GetMutable(const std::string& op_type) {
  auto it = op_utils_map_.find(op_type);
  if (it == op_utils_map_.end()) {
    return nullptr;
  } else {
    return &it->second;
  }
}

void OpUtilsMap::Insert(std::string op_type, OpUtils utils) {
  PADDLE_ENFORCE_NE(
      Contains(op_type),
      true,
      paddle::platform::errors::AlreadyExists(
          "Operator (%s)'s compatible utils has been registered.", op_type));
  op_utils_map_.insert({std::move(op_type), std::move(utils)});
}

}  // namespace pten
