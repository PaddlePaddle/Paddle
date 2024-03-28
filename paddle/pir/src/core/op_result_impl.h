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

#include "paddle/pir/src/core/value_impl.h"

namespace pir {
namespace detail {
///
/// \brief OpResultImpl is the implementation of an operation result.
///
class OpResultImpl : public ValueImpl {
 public:
  using ValueImpl::ValueImpl;

  static bool classof(const ValueImpl &value) {
    return value.kind() <= OUTLINE_RESULT_IDX;
  }

  ///
  /// \brief Get the parent operation of this result.(op_ptr = value_ptr +
  /// index)
  ///
  const Operation *owner() const;

  Operation *owner();

  ///
  /// \brief Get the result index of the operation result.
  ///
  uint32_t index() const;

  TEST_API ~OpResultImpl();

  ///
  /// \brief attribute related public interfaces
  ///
  Attribute attribute(const std::string &key) const;
  void set_attribute(const std::string &key, Attribute value);

  void *property(const std::string &key) const;
  void set_property(const std::string &key, const Property &value);

 private:
  int32_t ComputeOperationOffset() const;
};

///
/// \brief OpInlineResultImpl is the implementation of an operation result whose
/// index <= 5.
///
class OpInlineResultImpl : public OpResultImpl {
 public:
  TEST_API OpInlineResultImpl(Type type, uint32_t result_index);

  static bool classof(const ValueImpl &value) {
    return value.kind() < OUTLINE_RESULT_IDX;
  }

  uint32_t index() const { return kind(); }
};

///
/// \brief OpOutlineResultImpl is the implementation of an operation result
/// whose index > 5.
///
class OpOutlineResultImpl : public OpResultImpl {
 public:
  OpOutlineResultImpl(Type type, uint32_t outline_index)
      : OpResultImpl(type, OUTLINE_RESULT_IDX), outline_index_(outline_index) {}

  static bool classof(const ValueImpl &value) {
    return value.kind() == OUTLINE_RESULT_IDX;
  }

  uint32_t index() const { return outline_index_; }

 private:
  uint32_t outline_index_;
};

}  // namespace detail
}  // namespace pir
