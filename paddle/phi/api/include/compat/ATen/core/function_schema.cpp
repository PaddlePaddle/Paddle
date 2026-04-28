// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#include "ATen/core/function_schema.h"

namespace c10 {

namespace {

constexpr char kWildcardAliasSet[] = "*";

const char* schemaArgTypeName(SchemaArgType type) {
  if (type == SchemaArgType::input) {
    return "input";
  }
  if (type == SchemaArgType::output) {
    return "output";
  }
  return "unknown";
}

bool aliasSetsMayOverlap(const std::unordered_set<std::string>& lhs,
                         const std::unordered_set<std::string>& rhs) {
  if (lhs.empty() || rhs.empty()) {
    return false;
  }
  if (lhs.count(kWildcardAliasSet) > 0 || rhs.count(kWildcardAliasSet) > 0) {
    return true;
  }
  for (const auto& set : lhs) {
    if (rhs.count(set) > 0) {
      return true;
    }
  }
  return false;
}

const Argument& getSchemaArgumentOrThrow(const FunctionSchema& schema,
                                         const SchemaArgument& argument) {
  const auto& args = schema.getCorrectList(argument);
  TORCH_CHECK(argument.index < args.size(),
              "Schema ",
              schemaArgTypeName(argument.type),
              " index ",
              argument.index,
              " is out of bounds for size ",
              args.size());
  return args.at(argument.index);
}

bool aliasInfoMayContainAlias(const AliasInfo& lhs,
                              const AliasInfo& rhs,
                              bool bidirectional) {
  if (aliasSetsMayOverlap(lhs.afterSets(), rhs.afterSets())) {
    return true;
  }

  for (const auto& child : lhs.containedTypes()) {
    if (aliasInfoMayContainAlias(child, rhs, /*bidirectional=*/true)) {
      return true;
    }
  }

  if (!bidirectional) {
    return false;
  }
  for (const auto& child : rhs.containedTypes()) {
    if (aliasInfoMayContainAlias(lhs, child, /*bidirectional=*/true)) {
      return true;
    }
  }
  return false;
}

}  // namespace

std::ostream& operator<<(std::ostream& out, const Argument& arg) {
  out << arg.type()->str() << " " << arg.name();
  if (arg.default_value()) {
    out << " = " << arg.default_value();
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema) {
  out << "(";
  bool first = true;
  for (const auto& arg : schema.arguments()) {
    if (!first) {
      out << ", ";
    }
    out << arg;
    first = false;
  }
  if (schema.is_vararg()) {
    if (!first) {
      out << ", ";
    }
    out << "...";
  }
  out << ")";

  out << " -> ";

  if (schema.returns().size() == 1) {
    out << schema.returns()[0];
  } else {
    out << "(";
    first = true;
    for (const auto& ret : schema.returns()) {
      if (!first) {
        out << ", ";
      }
      out << ret;
      first = false;
    }
    out << ")";
  }

  return out;
}

std::optional<int> FunctionSchema::argumentIndexWithName(
    const std::string& name) const {
  for (size_t i = 0; i < arguments_.size(); ++i) {
    if (arguments_[i].name() == name) {
      return static_cast<int>(i);
    }
  }
  return std::nullopt;
}

const std::vector<Argument>& FunctionSchema::getCorrectList(
    const SchemaArgument& argument) const {
  if (argument.type == SchemaArgType::input) {
    return arguments();
  }
  if (argument.type == SchemaArgType::output) {
    return returns();
  }
  TORCH_INTERNAL_ASSERT(false, "Could not match argument type");
}

bool FunctionSchema::is_aliasing(const SchemaArgument& argument) const {
  const auto& arg = getSchemaArgumentOrThrow(*this, argument);
  return arg.alias_info() != nullptr;
}

bool FunctionSchema::is_mutable(const SchemaArgument& argument) const {
  const auto& arg = getSchemaArgumentOrThrow(*this, argument);
  return arg.alias_info() != nullptr && arg.alias_info()->isWrite();
}

bool FunctionSchema::is_mutable(const std::string& name) const {
  const auto index = argumentIndexWithName(name);
  TORCH_CHECK(
      index.has_value(), "Tried to test mutability of nonexistent name ", name);
  return is_mutable({SchemaArgType::input, static_cast<size_t>(*index)});
}

bool FunctionSchema::may_alias(const SchemaArgument& lhs,
                               const SchemaArgument& rhs) const {
  const auto& lhs_arg = getSchemaArgumentOrThrow(*this, lhs);
  const auto& rhs_arg = getSchemaArgumentOrThrow(*this, rhs);
  const auto* lhs_alias = lhs_arg.alias_info();
  const auto* rhs_alias = rhs_arg.alias_info();
  if (lhs_alias == nullptr || rhs_alias == nullptr) {
    return false;
  }
  return aliasSetsMayOverlap(lhs_alias->afterSets(), rhs_alias->afterSets());
}

bool FunctionSchema::may_contain_alias(const SchemaArgument& lhs,
                                       const SchemaArgument& rhs,
                                       bool bidirectional) const {
  const auto& lhs_arg = getSchemaArgumentOrThrow(*this, lhs);
  const auto& rhs_arg = getSchemaArgumentOrThrow(*this, rhs);
  const auto* lhs_alias = lhs_arg.alias_info();
  const auto* rhs_alias = rhs_arg.alias_info();
  if (lhs_alias == nullptr || rhs_alias == nullptr) {
    return false;
  }
  return aliasInfoMayContainAlias(*lhs_alias, *rhs_alias, bidirectional);
}

}  // namespace c10
