// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <glog/logging.h>
#include <llvm/ADT/ArrayRef.h>

#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "paddle/infrt/host_context/value.h"

namespace infrt {
namespace host_context {

/**
 * KernelFrame captures the states(input arguments, attributes, results)
 * associated with a kernel invocation.
 */
class KernelFrame {
 public:
  int GetNumArgs() const { return num_arguments_; }
  int GetNumResults() const {
    return value_or_attrs_.size() - num_arguments_ - GetNumAttributes();
  }
  int GetNumAttributes() const { return num_attrs_ == -1 ? 0 : num_attrs_; }

  //! Get something at a specific position \p index. The element might be an
  //! argument, an attribute or a result.
  template <typename T>
  T& GetElementAt(int index) {
    CHECK_LT(static_cast<size_t>(index), GetNumElements());
    return value_or_attrs_[index]->template get_or_default<T>();
  }

  Value* GetElementAt(int index) {
    CHECK_LT(static_cast<size_t>(index), GetNumElements());
    return value_or_attrs_[index];
  }

  // Get number of elements, either input, attributes or results.
  size_t GetNumElements() const { return value_or_attrs_.size(); }

  template <typename T>
  T& GetArgAt(int index) {
    CHECK_LT(index, GetNumArgs());
    return value_or_attrs_[index]->get<T>();
  }
  template <typename T>
  const T& GetArgAt(int index) const {
    CHECK_LT(index, GetNumArgs());
    return value_or_attrs_[index]->get<T>();
  }

  Value* GetArgAt(int index) {
    CHECK_LT(index, GetNumArgs());
    return value_or_attrs_[index];
  }

  // Get all arguments.
  llvm::ArrayRef<Value*> GetArguments() const {
    return GetValues(0, num_arguments_);
  }

  Value* GetAttributeAt(int idx) {
    // CHECK_NE(num_results_, -1)
    //<< "Must call SetNumResults before GetAttributeAt";
    CHECK_LT(idx, GetNumAttributes());
    return value_or_attrs_[num_arguments_ + idx];
  }

  void AddAttribute(Value* v) {
    CHECK_LE(num_results_, 0)
        << "Must call SetNumResults after calling AddAttribute";
    value_or_attrs_.emplace_back(v);
    if (num_attrs_ == -1) num_attrs_ = 0;
    num_attrs_++;

    CHECK_EQ(value_or_attrs_.size(),
             static_cast<size_t>(num_arguments_ + num_attrs_));
  }

  template <typename T, typename... Args>
  void EmplaceResult(Args&&... args) {
    EmplaceResult<T>(0, std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  void EmplaceResult(int index, Args&&... args) {
    SetResultAt(index, T(std::forward<Args>(args)...));
  }

  template <typename T>
  void SetResultAt(int index, T&& value) {
    CHECK_LT(index, GetNumResults()) << "Invalid result index";
    CHECK(value_or_attrs_[num_arguments_ + GetNumAttributes() + index]);
    value_or_attrs_[num_arguments_ + GetNumAttributes() + index]->set(
        std::move(value));
  }

  llvm::ArrayRef<Value*> GetResults() const {
    CHECK_GE(num_results_, 0) << "Invalid results num";
    return GetValues(num_arguments_ + GetNumAttributes(), num_results_);
  }
  llvm::MutableArrayRef<Value*> GetResults() {
    CHECK_GE(num_results_, 0) << "Invalid results num";
    return GetMutableValues(num_arguments_ + GetNumAttributes(), num_results_);
  }

  llvm::ArrayRef<Value*> GetValues(size_t from, size_t length) const {
    CHECK_LE(from + length, GetNumElements());
    if (length == 0) return {};

    return llvm::makeArrayRef(&value_or_attrs_[from], length);
  }

  llvm::MutableArrayRef<Value*> GetMutableValues(size_t from, size_t length) {
    CHECK_LE(from + length, GetNumElements());
    if (length == 0) return {};
    return llvm::makeMutableArrayRef(&value_or_attrs_[from], length);
  }

#ifndef NDEBUG
  std::string DumpArgTypes() const;
#endif

  bool IsEmpty() const { return value_or_attrs_.empty(); }

 protected:
  int num_arguments_{};
  int num_attrs_{-1};
  int num_results_{-1};

  llvm::SmallVector<Value*, 8> value_or_attrs_;
};

std::ostream& operator<<(std::ostream& os, const KernelFrame& frame);

class KernelFrameBuilder : public KernelFrame {
 public:
  void AddArgument(Value* value) {
    CHECK(value);
    CHECK_EQ(num_attrs_, -1)
        << "Should call AddArgument before calling SetAttributes";
    value_or_attrs_.push_back(value);
    ++num_arguments_;
  }

  void SetResults(llvm::ArrayRef<Value*> values) {
    CHECK_EQ(num_arguments_ + GetNumAttributes(),
             static_cast<int>(value_or_attrs_.size()));
    for (Value* x : values) {
      value_or_attrs_.push_back(x);
    }
    num_results_ = values.size();
  }

  void SetNumResults(size_t n) {
    CHECK_EQ(num_arguments_ + GetNumAttributes(),
             static_cast<int>(value_or_attrs_.size()));
    for (size_t i = 0; i < n; i++) {
      value_or_attrs_.emplace_back(new Value);
    }
    num_results_ = n;
  }

  void SetResultAt(int result_id, Value* value) {
    CHECK_EQ(static_cast<int>(value_or_attrs_.size()),
             num_arguments_ + GetNumAttributes() + num_results_)
        << "Call SetNumResults first";
    CHECK_LT(result_id + num_arguments_ + GetNumAttributes(),
             static_cast<int>(value_or_attrs_.size()));
    CHECK(value);
    value_or_attrs_[num_arguments_ + GetNumAttributes() + result_id]->set(
        value);
  }

  void Reset() {
    value_or_attrs_.clear();
    num_arguments_ = 0;
    num_results_ = -1;
    num_attrs_ = -1;
  }
};

}  // namespace host_context
}  // namespace infrt
