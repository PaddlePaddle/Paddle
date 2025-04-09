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

namespace phi::distributed {

InferSpmdContext::InferSpmdContext(
    paddle::small_vector<DistMetaTensor, phi::kInputSmallVectorSize> inputs,
    paddle::small_vector<Attribute, phi::kAttrSmallVectorSize> attrs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    EmplaceBackInput(inputs[i]);
  }
  for (size_t i = 0; i < attrs.size(); i++) {
    EmplaceBackAttr(attrs[i]);
  }
}

void InferSpmdContext::EmplaceBackInput(DistMetaTensor input) {
  int index = static_cast<int>(inputs_.size());
  inputs_.emplace_back(std::move(input));
  input_range_.emplace_back(std::pair<int, int>(index, index + 1));
}

void InferSpmdContext::EmplaceBackInputs(
    paddle::small_vector<DistMetaTensor, phi::kInputSmallVectorSize> inputs) {
  int index = static_cast<int>(inputs_.size());
  input_range_.emplace_back(std::pair<int, int>(index, index + inputs.size()));
  inputs_.insert(inputs_.end(),
                 std::make_move_iterator(inputs.begin()),
                 std::make_move_iterator(inputs.end()));
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
    PADDLE_THROW(common::errors::InvalidArgument(
        "Attribute cast error in InferSpmd Context, the input attr type is "
        "`%s`, but the expected attribute type is `%s`.",
        attrs_.at(idx).type().name(),
        std::type_index(typeid(AttrType)).name()));
  }
}

template float InferSpmdContext::AttrAt(size_t idx) const;
template int InferSpmdContext::AttrAt(size_t idx) const;
template int64_t InferSpmdContext::AttrAt(size_t idx) const;

template <>
bool InferSpmdContext::AttrAt(size_t idx) const {
  try {
    auto attr = attrs_.at(idx);
    if (attr.type() == typeid(int)) {
      return static_cast<bool>(paddle::get<int>(attr));
    } else {
      return paddle::get<bool>(attr);
    }
  } catch (paddle::bad_variant_access const& e) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Attribute cast error in InferSpmd Context, the input attr type is "
        "`%s`, but the expected attribute type is `bool`.",
        attrs_.at(idx).type().name()));
  }
}

template <>
std::vector<int> InferSpmdContext::AttrAt(size_t idx) const {
  try {
    auto attr = attrs_.at(idx);
    if (attr.type() == typeid(std::vector<bool>)) {
      std::vector<bool> val = PADDLE_GET_CONST(std::vector<bool>, attr);
      return std::vector<int>(val.begin(), val.end());
    } else {
      return paddle::get<std::vector<int>>(attr);
    }
  } catch (paddle::bad_variant_access const& e) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Attribute cast error in InferSpmd Context, the input attr type is "
        "`%s`, but the expected attribute type is `std::vector<int>`.",
        attrs_.at(idx).type().name()));
  }
}

template <>
std::vector<int64_t> InferSpmdContext::AttrAt(size_t idx) const {
  try {
    auto attr = attrs_.at(idx);
    if (attr.type() == typeid(std::vector<bool>)) {
      std::vector<bool> val = PADDLE_GET_CONST(std::vector<bool>, attr);
      return std::vector<int64_t>(val.begin(), val.end());
    } else if (attr.type() == typeid(std::vector<int>)) {
      std::vector<int> val = PADDLE_GET_CONST(std::vector<int>, attr);
      return std::vector<int64_t>(val.begin(), val.end());
    } else {
      return PADDLE_GET_CONST(std::vector<int64_t>, attr);
    }
  } catch (paddle::bad_variant_access const& e) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Attribute cast error in InferSpmd Context, the input attr type is "
        "`%s`, but the expected attribute type is `std::vector<int64_t>`.",
        attrs_.at(idx).type().name()));
  }
}

const Attribute& InferSpmdContext::AttrAt(size_t idx) const {
  return attrs_.at(idx);
}

const std::pair<int, int>& InferSpmdContext::InputRangeAt(size_t idx) const {
  return input_range_.at(idx);
}

const std::vector<const DistMetaTensor*> InferSpmdContext::InputsBetween(
    size_t start, size_t end) const {
  std::vector<const DistMetaTensor*> result;
  result.reserve(end - start);
  for (size_t i = start; i < end; ++i) {
    auto& in = inputs_.at(i);
    result.emplace_back(&in);
    // result.emplace_back(in.initialized() ? &in : nullptr);
  }

  return result;
}

SpmdRuleFactory& SpmdRuleFactory::Instance() {
  static SpmdRuleFactory g_spmd_rule_map;
  return g_spmd_rule_map;
}

bool SpmdRuleFactory::ContainsSpmdRule(const std::string& kernel_name) const {
  return spmd_rule_map_.count(kernel_name) > 0;
}

int SpmdRuleFactory::InsertSpmdRule(std::string kernel_name, SpmdRule rule) {
  spmd_rule_map_.insert({std::move(kernel_name), rule});
  return 0;
}

const SpmdRule& SpmdRuleFactory::GetSpmdRule(
    const std::string& kernel_name) const {
  auto it = spmd_rule_map_.find(kernel_name);
  PADDLE_ENFORCE_NE(
      it,
      spmd_rule_map_.end(),
      common::errors::NotFound("`%s` Kernel's Spmd rules is not registered.",
                               kernel_name));
  return it->second;
}

}  // namespace phi::distributed
