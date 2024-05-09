// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#pragma once

#include <functional>
#include <ostream>
#include <variant>

namespace cinn {
namespace common {

struct Language_Unknown {};

struct Language_Host {};

struct Language_CUDA {};

/**
 * The architecture used by the target. Determines the instruction set to use.
 */
using LanguageBase =
    std::variant<Language_Unknown, Language_Host, Language_CUDA>;
struct Language final : public LanguageBase {
  using LanguageBase::LanguageBase;

  template <typename VisitorT>
  decltype(auto) Visit(VisitorT&& visitor) const {
    return std::visit(visitor, variant());
  }

  const LanguageBase& variant() const {
    return static_cast<const LanguageBase&>(*this);
  }

  bool operator==(const auto& other) const {
    return this->index() == other.index();
  }

  bool operator!=(const auto& other) const { return !(*this == other); }
};

inline bool IsDefined(Language Language) {
  return !std::holds_alternative<Language_Unknown>(Language);
}

}  // namespace common
}  // namespace cinn

namespace std {

template <>
struct hash<::cinn::common::Language> {
  std::size_t operator()(const ::cinn::common::Language& Language) const {
    return Language.index();
  }
};

}  // namespace std
