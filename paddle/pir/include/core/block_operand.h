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

#include "paddle/pir/include/core/cast_utils.h"
#include "paddle/pir/include/core/type.h"

namespace pir {
class Operation;
class Value;
class Block;

namespace detail {
class BlockOperandImpl;
}  // namespace detail

///
/// \brief OpOperand class represents the op_operand of operation. This class
/// only provides interfaces, for specific implementation, see Impl class.
///
class IR_API BlockOperand {
 public:
  BlockOperand() = default;

  BlockOperand(const BlockOperand &other) = default;

  BlockOperand(detail::BlockOperandImpl *impl) : impl_(impl) {}  // NOLINT

  BlockOperand &operator=(const BlockOperand &rhs);

  bool operator==(const BlockOperand &other) const {
    return impl_ == other.impl_;
  }

  bool operator!=(const BlockOperand &other) const {
    return !operator==(other);
  }

  bool operator!() const { return impl_ == nullptr; }

  operator bool() const;

  BlockOperand next_use() const;

  Block *source() const;

  void set_source(Block *source);

  Operation *owner() const;

  void RemoveFromUdChain();

  friend Operation;

  detail::BlockOperandImpl *impl() const { return impl_; }

 private:
  detail::BlockOperandImpl *impl_{nullptr};
};

}  // namespace pir
