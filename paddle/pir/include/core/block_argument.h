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

#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/core/value.h"

namespace pir {
class Block;

namespace detail {
class BlockArgumentImpl;
}  // namespace detail

///
/// \brief BlockArgument class represents the value defined by a argument of
/// block. This class only provides interfaces, for specific implementation,
/// see Impl class.
///
class IR_API BlockArgument : public Value {
 public:
  BlockArgument() = default;
  Block *owner() const;
  uint32_t index() const;
  const std::string &keyword() const;
  bool is_kwarg() const;

  const AttributeMap &attributes() const;
  Attribute attribute(const std::string &key) const;
  void set_attribute(const std::string &key, Attribute value);

 private:
  /// constructor
  BlockArgument(detail::BlockArgumentImpl *impl);  // NOLINT

  /// create a new argument with the given type and owner.
  static BlockArgument Create(Type type, Block *owner, uint32_t index);
  static BlockArgument Create(Type type,
                              Block *owner,
                              const std::string &keyword);
  /// Destroy the argument.
  void Destroy();
  /// set the position in the block argument list.
  void set_index(uint32_t index);
  // Access create annd destroy.
  friend Block;

  // Access classof annd dyn_cast_from.
  friend Value;
  static bool classof(Value value);
  static BlockArgument dyn_cast_from(Value value);
};
}  // namespace pir
