// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include "paddle/pir/include/core/dll_decl.h"

namespace pir {
class Operation;
class Value;
class Type;

namespace detail {
class OpOperandImpl;
}  // namespace detail

///
/// \brief OpOperand class represents the op_operand of operation. This class
/// only provides interfaces, for specific implementation, see Impl class.
///
class IR_API OpOperand {
 public:
  OpOperand() = default;

  OpOperand(const OpOperand &other) = default;

  OpOperand(detail::OpOperandImpl *impl) : impl_(impl) {}  // NOLINT

  OpOperand &operator=(const OpOperand &rhs);

  bool operator==(const OpOperand &other) const { return impl_ == other.impl_; }

  bool operator!=(const OpOperand &other) const { return !operator==(other); }

  bool operator!() const { return impl_ == nullptr; }

  operator bool() const;

  OpOperand next_use() const;

  Value source() const;

  Type type() const;

  void set_source(Value value);

  Operation *owner() const;

  uint32_t index() const;

  void RemoveFromUdChain();

  friend Operation;

 private:
  detail::OpOperandImpl *impl_{nullptr};
};
}  // namespace pir
