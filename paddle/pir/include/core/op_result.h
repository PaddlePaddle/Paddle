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

#include "paddle/pir/include/core/value.h"

namespace pir {

namespace detail {
class OpResultImpl;
}  // namespace detail

///
/// \brief OpResult class represents the value defined by a result of operation.
/// This class only provides interfaces, for specific implementation, see Impl
/// class.
///
class IR_API OpResult : public Value {
 public:
  OpResult(std::nullptr_t ptr = nullptr) : Value(ptr){};  // NOLINT
  Operation *owner() const;
  // Return the result index of this op result.
  uint32_t index() const;
  bool operator==(const OpResult &other) const;

  Attribute attribute(const std::string &key) const;
  void set_attribute(const std::string &key, Attribute value);

  void *property(const std::string &key) const;
  void set_property(const std::string &key, const Property &value);

 private:
  friend Operation;
  OpResult(detail::OpResultImpl *impl);  // NOLINT
  // Access classof and dyn_cast_from.
  friend Value;
  friend struct std::hash<OpResult>;
  static bool classof(Value value);
  static OpResult dyn_cast_from(Value value);
};

}  // namespace pir

namespace std {
template <>
struct hash<pir::OpResult> {
  std::size_t operator()(const pir::OpResult &obj) const {
    return std::hash<pir::Value>()(obj);
  }
};
}  // namespace std
