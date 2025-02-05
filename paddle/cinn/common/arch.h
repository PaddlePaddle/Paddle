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

#define CINN_ARCH_CLASS_NAMES(_macro)                                       \
  _macro(X86Arch) _macro(ARMArch) _macro(NVGPUArch) _macro(HygonDCUArchHIP) \
      _macro(HygonDCUArchSYCL)

#define DEFINE_CINN_ARCH(class_name) \
  struct class_name {};
CINN_ARCH_CLASS_NAMES(DEFINE_CINN_ARCH);
#undef DEFINE_CINN_ARCH

/**
 * The architecture used by the target. Determines the instruction set to use.
 */
using ArchBase = std::variant<
#define LIST_CINN_ARCH_ALTERNATIVE(class_name) class_name,
    CINN_ARCH_CLASS_NAMES(LIST_CINN_ARCH_ALTERNATIVE)
#undef LIST_CINN_ARCH_ALTERNATIVE
        UnknownArch>;
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
