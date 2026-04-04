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

#pragma once

#include <ostream>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace c10 {
/**
 * class AliasInfo
 *
 * Data structure to hold aliasing information for an `Argument`. They can be
 * nested to represent aliasing information on contained types.
 *
 * There is a `beforeSet` which describes the aliasing information before the
 * operator executes, and an `afterSet` that describes aliasing info
 * after execution.
 */
class AliasInfo {
 public:
  AliasInfo() = default;
  AliasInfo(bool is_write,
            const std::set<std::string>& before_qual_strings,
            const std::set<std::string>& after_qual_strings)
      : isWrite_(is_write) {
    for (const auto& s : before_qual_strings) {
      beforeSets_.insert(s);
    }
    for (const auto& s : after_qual_strings) {
      afterSets_.insert(s);
    }
  }

  bool isWrite() const { return isWrite_; }

  const std::unordered_set<std::string>& beforeSets() const {
    return beforeSets_;
  }

  const std::unordered_set<std::string>& afterSets() const {
    return afterSets_;
  }

  // the alias info for the contained types of the type
  // e.g. if this is an annotation on List[T], `sets` refers to
  // the alias sets that the list may be in
  // while containedTypes()[0] refers to the sets that members of the list
  // may be in
  void addContainedType(AliasInfo aliasInfo) {
    containedTypes_.push_back(std::move(aliasInfo));
  }
  const std::vector<AliasInfo>& containedTypes() const {
    return containedTypes_;
  }

 private:
  std::unordered_set<std::string> beforeSets_;
  std::unordered_set<std::string> afterSets_;
  std::vector<AliasInfo> containedTypes_;
  bool isWrite_ = false;
};

inline bool operator==(const AliasInfo& lhs, const AliasInfo& rhs) {
  return lhs.isWrite() == rhs.isWrite() &&
         lhs.beforeSets() == rhs.beforeSets() &&
         lhs.afterSets() == rhs.afterSets() &&
         lhs.containedTypes() == rhs.containedTypes();
}

// this does match the way things are represented in the schema
inline std::ostream& operator<<(std::ostream& out, const AliasInfo& aliasInfo) {
  out << '(';
  bool first = true;
  for (const auto& set : aliasInfo.beforeSets()) {
    if (first) {
      first = false;
    } else {
      out << '|';
    }
    out << set;
  }
  if (aliasInfo.isWrite()) {
    out << '!';
  }
  if (aliasInfo.beforeSets() != aliasInfo.afterSets()) {
    out << " -> ";
    first = true;
    for (const auto& set : aliasInfo.afterSets()) {
      if (first) {
        first = false;
      } else {
        out << '|';
      }
      out << set;
    }
  }
  out << ')';
  return out;
}
}  // namespace c10

inline std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
  lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
  return lhs;
}

namespace std {
template <>
struct hash<c10::AliasInfo> {
  size_t operator()(const c10::AliasInfo& aliasInfo) const {
    auto hash = std::hash<bool>()(aliasInfo.isWrite());

    // NOTE: for unordered_set hashes, we couldn't use hash_combine
    // because hash_combine is order dependent. Instead, we choose to
    // use XOR as the combining function as XOR is commutative.
    size_t before_set_hash_seed = 0;
    for (auto& e : aliasInfo.beforeSets()) {
      auto symbol_hash = std::hash<std::string>()(e);
      before_set_hash_seed = before_set_hash_seed ^ symbol_hash;
    }
    size_t after_set_hash_seed = 0;
    for (auto& e : aliasInfo.afterSets()) {
      auto symbol_hash = std::hash<std::string>()(e);
      after_set_hash_seed = after_set_hash_seed ^ symbol_hash;
    }

    hash = hash_combine(hash, before_set_hash_seed);
    hash = hash_combine(hash, after_set_hash_seed);
    for (auto& e : aliasInfo.containedTypes()) {
      auto contained_type_hash = std::hash<c10::AliasInfo>()(e);
      hash = hash_combine(hash, contained_type_hash);
    }
    return hash;
  }
};
}  // namespace std
