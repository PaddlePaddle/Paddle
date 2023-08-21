/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"

namespace phi {
namespace distributed {

void InferSpmdContext::EmplaceBackInput(MetaTensor input) {
  inputs_.emplace_back(std::move(input));
}

void InferSpmdContext::EmplaceBackAttr(Attribute attr) {
  attrs_.emplace_back(std::move(attr));
}

const MetaTensor& InferSpmdContext::InputAt(size_t idx) const {
  return inputs_.at(idx);
}

template <typename AttrType>
const AttrType& InferSpmdContext::AttrAt(size_t idx) const {
  try {
    return paddle::get<AttrType>(attrs_.at(idx));
  } catch (paddle::bad_variant_access const& e) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Attribute cast error in InferSpmd Context, the expected attribute "
        "type is `%s`.",
        std::type_index(typeid(AttrType)).name()));
  }
}

const Attribute& InferSpmdContext::AttrAt(size_t idx) const {
  return attrs_.at(idx);
}

template const bool& InferSpmdContext::AttrAt(size_t idx) const;

SpmdRuleFactory& SpmdRuleFactory::Instance() {
  static SpmdRuleFactory g_spmd_rule_map;
  return g_spmd_rule_map;
}

bool SpmdRuleFactory::ContainsInferSpmdFn(
    const std::string& kernel_name) const {
  return infer_spmd_fn_map_.count(kernel_name) > 0;
}

void SpmdRuleFactory::InsertInferSpmdFn(std::string kernel_name,
                                        InferSpmdFn fn) {
  PADDLE_ENFORCE_NE(
      ContainsInferSpmdFn(kernel_name),
      true,
      phi::errors::AlreadyExists(
          "`%s` Kernel's InferSpmdFn has been registered.", kernel_name));
  infer_spmd_fn_map_.insert({std::move(kernel_name), std::move(fn)});
}

const InferSpmdFn& SpmdRuleFactory::GetInferSpmdFn(
    const std::string& kernel_name) const {
  auto it = infer_spmd_fn_map_.find(kernel_name);
  PADDLE_ENFORCE_NE(
      it,
      infer_spmd_fn_map_.end(),
      phi::errors::NotFound("`%s` Kernel's InferSpmdFn is not registered.",
                            kernel_name));
  return it->second;
}

}  // namespace distributed
}  // namespace phi
