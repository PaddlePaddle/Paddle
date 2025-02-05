// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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
#include <optional>
#include <vector>
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"

namespace cinn {
namespace common {

template <typename Op>
inline std::optional<ir::Expr> TryConstFold(ir::Expr a, ir::Expr b);

template <>
inline std::optional<ir::Expr> TryConstFold<ir::Add>(ir::Expr a, ir::Expr b) {
  const ir::IntImm* pa = a.As<ir::IntImm>();
  const ir::IntImm* pb = b.As<ir::IntImm>();
  const auto& rtype = a.type();
  if (pa && pb) {
    int64_t res = pa->value + pb->value;
    return cinn::common::make_shared<ir::IntImm>(rtype, res);
  }
  if (pa && pa->value == 0) return b;
  if (pb && pb->value == 0) return a;
  return std::nullopt;
}

template <>
inline std::optional<ir::Expr> TryConstFold<ir::Sub>(ir::Expr a, ir::Expr b) {
  const ir::IntImm* pa = a.As<ir::IntImm>();
  const ir::IntImm* pb = b.As<ir::IntImm>();
  const auto& rtype = a.type();
  if (pa && pb) {
    int64_t res = pa->value - pb->value;
    return cinn::common::make_shared<ir::IntImm>(rtype, res);
  }
  if (pb && pb->value == 0) return a;
  return std::nullopt;
}

template <>
inline std::optional<ir::Expr> TryConstFold<ir::Mul>(ir::Expr a, ir::Expr b) {
  const ir::IntImm* pa = a.As<ir::IntImm>();
  const ir::IntImm* pb = b.As<ir::IntImm>();
  const auto& rtype = a.type();
  if (pa && pb) {
    int64_t res = pa->value * pb->value;
    return cinn::common::make_shared<ir::IntImm>(rtype, res);
  }
  if (pa) {
    if (pa->value == 1) return b;
    if (pa->value == 0) return a;
  }
  if (pb) {
    if (pb->value == 1) return a;
    if (pb->value == 0) return b;
  }
  return std::nullopt;
}

template <>
inline std::optional<ir::Expr> TryConstFold<ir::Div>(ir::Expr a, ir::Expr b) {
  const ir::IntImm* pa = a.As<ir::IntImm>();
  const ir::IntImm* pb = b.As<ir::IntImm>();
  const auto& rtype = a.type();
  if (pa && pb) {
    int64_t res = pa->value / pb->value;
    return cinn::common::make_shared<ir::IntImm>(rtype, res);
  }
  if (pa) {
    if (pa->value == 0) return a;
  }
  if (pb) {
    if (pb->value == 1) return a;
  }
  return std::nullopt;
}

template <>
inline std::optional<ir::Expr> TryConstFold<ir::Mod>(ir::Expr a, ir::Expr b) {
  const ir::IntImm* pa = a.As<ir::IntImm>();
  const ir::IntImm* pb = b.As<ir::IntImm>();
  const auto& rtype = a.type();
  if (pa && pb) {
    int64_t res = pa->value % pb->value;
    return cinn::common::make_shared<ir::IntImm>(rtype, res);
  }
  if (pa) {
    if (pa->value == 0) return a;
  }
  if (pb) {
    if (pb->value == 1) return ir::Zero(rtype);
  }
  return std::nullopt;
}

template <>
inline std::optional<ir::Expr> TryConstFold<ir::Min>(ir::Expr a, ir::Expr b) {
  const ir::IntImm* pa = a.As<ir::IntImm>();
  const ir::IntImm* pb = b.As<ir::IntImm>();
  const auto& rtype = a.type();
  if (pa && pb) {
    int64_t res = std::min(pa->value, pb->value);
    return cinn::common::make_shared<ir::IntImm>(rtype, res);
  }
  return std::nullopt;
}

template <>
inline std::optional<ir::Expr> TryConstFold<ir::Max>(ir::Expr a, ir::Expr b) {
  const ir::IntImm* pa = a.As<ir::IntImm>();
  const ir::IntImm* pb = b.As<ir::IntImm>();
  const auto& rtype = a.type();
  if (pa && pb) {
    int64_t res = std::max(pa->value, pb->value);
    return cinn::common::make_shared<ir::IntImm>(rtype, res);
  }
  return std::nullopt;
}
}  // namespace common
}  // namespace cinn
