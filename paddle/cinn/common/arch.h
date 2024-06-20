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
#include "paddle/common/overloaded.h"

namespace cinn {
namespace common {

struct UnknownArch {};

struct X86Arch {};

struct ARMArch {};

struct NVGPUArch {};

/**
 * The architecture used by the target. Determines the instruction set to use.
 */
using ArchBase = std::variant<UnknownArch, X86Arch, ARMArch, NVGPUArch>;
struct Arch final : public ArchBase {
  using ArchBase::ArchBase;

  template <typename VisitorT>
  decltype(auto) Visit(VisitorT&& visitor) const {
    return std::visit(visitor, variant());
  }

  const ArchBase& variant() const {
    return static_cast<const ArchBase&>(*this);
  }

  DEFINE_MATCH_METHOD();

  bool operator==(const auto& other) const {
    return this->index() == other.index();
  }

  bool operator!=(const auto& other) const { return !(*this == other); }
};

inline bool IsDefined(Arch arch) {
  return !std::holds_alternative<UnknownArch>(arch);
}

}  // namespace common
}  // namespace cinn

namespace std {

template <>
struct hash<::cinn::common::Arch> {
  std::size_t operator()(const ::cinn::common::Arch& arch) const {
    return arch.index();
  }
};

}  // namespace std
