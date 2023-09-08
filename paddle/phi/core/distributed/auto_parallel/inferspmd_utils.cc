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

void InferSpmdContext::EmplaceBackInput(DistMetaTensor input) {
  inputs_.emplace_back(std::move(input));
}

void InferSpmdContext::EmplaceBackAttr(Attribute attr) {
  attrs_.emplace_back(std::move(attr));
}

const DistMetaTensor& InferSpmdContext::InputAt(size_t idx) const {
  return inputs_.at(idx);
}

template <typename AttrType>
AttrType InferSpmdContext::AttrAt(size_t idx) const {
  try {
    return paddle::get<AttrType>(attrs_.at(idx));
  } catch (paddle::bad_variant_access const& e) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Attribute cast error in InferSpmd Context, the input attr type is "
        "`%s`, but the expected attribute type is `%s`.",
        attrs_.at(idx).type().name(),
        std::type_index(typeid(AttrType)).name()));
  }
}

template <>
bool InferSpmdContext::AttrAt<bool>(size_t idx) const {
  try {
    auto attr = attrs_.at(idx);
    if (attr.type() == typeid(int)) {
      return static_cast<bool>(paddle::get<int>(attr));
    } else {
      return paddle::get<bool>(attr);
    }
  } catch (paddle::bad_variant_access const& e) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Attribute cast error in InferSpmd Context, the input attr type is "
        "`%s`, but the expected attribute type is `bool`.",
        attrs_.at(idx).type().name()));
  }
}

const Attribute& InferSpmdContext::AttrAt(size_t idx) const {
  return attrs_.at(idx);
}

SpmdRuleFactory& SpmdRuleFactory::Instance() {
  static SpmdRuleFactory g_spmd_rule_map;
  return g_spmd_rule_map;
}

bool SpmdRuleFactory::ContainsSpmdRule(const std::string& kernel_name) const {
  return spmd_rule_map_.count(kernel_name) > 0;
}

int SpmdRuleFactory::InsertSpmdRule(std::string kernel_name, SpmdRule rule) {
  spmd_rule_map_.insert({std::move(kernel_name), std::move(rule)});
  return 0;
}

const SpmdRule& SpmdRuleFactory::GetSpmdRule(
    const std::string& kernel_name) const {
  auto it = spmd_rule_map_.find(kernel_name);
  PADDLE_ENFORCE_NE(
      it,
      spmd_rule_map_.end(),
      phi::errors::NotFound("`%s` Kernel's Spmd rules is not registered.",
                            kernel_name));
  return it->second;
}

}  // namespace distributed
}  // namespace phi
