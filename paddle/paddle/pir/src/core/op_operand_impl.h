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
#include "paddle/pir/include/core/value.h"

namespace pir {

class Operation;

namespace detail {
///
/// \brief OpOperandImpl
///
class OpOperandImpl {
 public:
  Operation *owner() const;

  OpOperandImpl *next_use();

  Value source() const;

  void set_source(Value value);

  uint32_t index() const;

  /// Remove this op_operand from the current use list.
  void RemoveFromUdChain();

  ~OpOperandImpl();

  friend Operation;

 private:
  OpOperandImpl(Value source, Operation *owner);

  // Insert self to the UD chain held by source_;
  // It is not safe. So set private.
  void InsertToUdChain();

  Value source_;

  OpOperandImpl *next_use_ = nullptr;

  OpOperandImpl **prev_use_addr_ = nullptr;

  Operation *const owner_ = nullptr;
};

}  // namespace detail
}  // namespace pir
