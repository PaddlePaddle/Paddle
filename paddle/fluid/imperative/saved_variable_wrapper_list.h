// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <utility>
#include <vector>

namespace paddle {
namespace imperative {

class VariableWrapper;

class SavedVariableWrapperList {
 public:
  SavedVariableWrapperList() : vars_(), is_grad_(false) {}

  template <typename... Args>
  explicit SavedVariableWrapperList(bool is_grad, Args&&... args)
      : vars_(std::forward<Args>(args)...), is_grad_(is_grad) {}

  bool IsGrad() const { return is_grad_; }

  void SetIsGrad(bool is_grad) { is_grad_ = is_grad; }

  const std::vector<std::shared_ptr<VariableWrapper>>& VarList() const {
    return vars_;
  }

  std::vector<std::shared_ptr<VariableWrapper>>* MutableVarList() {
    return &vars_;
  }

  /* Borrow method from std::vector */
  size_t size() const { return vars_.size(); }

  bool empty() const { return vars_.empty(); }

  template <typename... ARGS>
  void emplace_back(ARGS&&... args) {
    vars_.emplace_back(std::forward<ARGS>(args)...);
  }

  using Iterator = std::vector<std::shared_ptr<VariableWrapper>>::iterator;

  using ConstIterator =
      std::vector<std::shared_ptr<VariableWrapper>>::const_iterator;

  Iterator begin() { return vars_.begin(); }

  Iterator end() { return vars_.end(); }

  ConstIterator begin() const { return vars_.begin(); }

  ConstIterator end() const { return vars_.end(); }

  std::shared_ptr<VariableWrapper>& operator[](size_t idx) {
    return vars_[idx];
  }

  const std::shared_ptr<VariableWrapper>& operator[](size_t idx) const {
    return vars_[idx];
  }

  operator const std::vector<std::shared_ptr<VariableWrapper>>&() const {
    return vars_;
  }

 private:
  std::vector<std::shared_ptr<VariableWrapper>> vars_;
  bool is_grad_;
};

}  // namespace imperative
}  // namespace paddle
