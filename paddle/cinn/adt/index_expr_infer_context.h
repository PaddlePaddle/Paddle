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

#include <unordered_map>

#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/m_expr.h"

namespace cinn::adt::equation {

class IndexExprInferContext final {
 public:
  IndexExprInferContext(const IndexExprInferContext&) = delete;
  IndexExprInferContext(IndexExprInferContext&&) = delete;

  IndexExprInferContext() = default;

  const Value& GetValue(const Variable& variable) const {
    return map_.at(variable);
  }

  auto SetValue(const Variable& variable, const Value& value) {
    return map_.emplace(variable, value);
  }

  bool HasValue(const Variable& variable) const {
    return map_.count(variable) > 0;
  }

  void AddTensorIndex2Tensor(const Index& index,
                             const m_expr::Tensor& tensor_ptr) {
    CHECK(tensor_index2tensor_.emplace(index, tensor_ptr).second);
    CHECK(tensor2tensor_index_.emplace(tensor_ptr, index).second);
  }

  const m_expr::Tensor& GetTensor(const Index& index) const {
    return tensor_index2tensor_.at(index);
  }

  const Index GetIndex(const m_expr::Tensor& tensor) const {
    return tensor2tensor_index_.at(tensor);
  }

 private:
  std::unordered_map<const Variable, Value> map_;
  std::unordered_map<const Index, const m_expr::Tensor> tensor_index2tensor_;
  std::unordered_map<const m_expr::Tensor, const Index> tensor2tensor_index_;
};

}  // namespace cinn::adt::equation
