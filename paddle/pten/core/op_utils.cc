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

KernelSignatureMap* KernelSignatureMap::kernel_signature_map_ = nullptr;
std::once_flag KernelSignatureMap::init_flag_;

OpUtils& OpUtils::Instance() {
  static OpUtils g_op_utils;
  return g_op_utils;
}

bool OpUtils::Contains(const std::string& op_type) const {
  return args_fn_map_.count(op_type);
}

void OpUtils::InsertArgumentMappingFn(const std::string& op_type,
                                      ArgumentMappingFn fn) {
  PADDLE_ENFORCE_NE(
      Contains(op_type),
      true,
      paddle::platform::errors::AlreadyExists(
          "Operator (%s)'s compatible utils has been registered.", op_type));
  args_fn_map_.insert({op_type, std::move(fn)});
}

void OpUtils::InsertInferMetaFn(const std::string& kernel_name_prefix,
                                InferMetaFn fn) {
  PADDLE_ENFORCE_EQ(
      infer_meta_fn_map_.count(kernel_name_prefix),
      0UL,
      paddle::platform::errors::AlreadyExists(
          "Series kernel (%s)'s infermeta function has been registered.",
          kernel_name_prefix));
  infer_meta_fn_map_.insert({kernel_name_prefix, std::move(fn)});
}

ArgumentMappingFn OpUtils::GetArgumentMappingFn(
    const std::string& op_type) const {
  auto it = args_fn_map_.find(op_type);
  if (it == args_fn_map_.end()) {
    auto func =
        [op_type](const ArgumentMappingContext& ctx) -> KernelSignature {
      return KernelSignatureMap::Instance().Get(TransToPtenKernelName(op_type));
    };
    return func;
  } else {
    return it->second;
  }
}

InferMetaFn OpUtils::GetInferMetaFn(
    const std::string& kernel_name_prefix) const {
  auto it = infer_meta_fn_map_.find(kernel_name_prefix);
  PADDLE_ENFORCE_NE(it,
                    infer_meta_fn_map_.end(),
                    paddle::platform::errors::NotFound(
                        "Cannot found Series kernel (%s)'s infermeta function.",
                        kernel_name_prefix));
  return it->second;
}

}  // namespace pten
