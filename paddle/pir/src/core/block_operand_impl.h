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

#include "paddle/pir/include/core/block_operand.h"

namespace pir {
class Operation;
class Block;

namespace detail {
///
/// \brief OpOperandImpl
///
class BlockOperandImpl {
 public:
  Operation* owner() const;

  BlockOperand next_use() const;

  Block* source() const;

  void set_source(Block*);

  /// Remove this op_operand from the current use list.
  void RemoveFromUdChain();

  ~BlockOperandImpl();

  friend Operation;

 private:
  BlockOperandImpl(Block* source, Operation* owner);

  // Insert self to the UD chain held by source_;
  // It is not safe. So set private.
  void InsertToUdChain();

  BlockOperand next_use_ = nullptr;

  BlockOperand* prev_use_addr_ = nullptr;

  Block* source_;

  Operation* const owner_ = nullptr;
};

}  // namespace detail
}  // namespace pir
