// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
#include <vector>

#include "glog/logging.h"
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/dim_expr.h"
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/utils/type_defs.h"

namespace cinn::adt::config {

using DimTuple = List<DimExpr>;

class OpEquationContext {
 public:
  OpEquationContext(const OpEquationContext&) = delete;
  OpEquationContext(OpEquationContext&&) = delete;
  virtual ~OpEquationContext() {}

  virtual const std::vector<std::uint64_t>& GetInTensorsRanks() const = 0;

  virtual const std::vector<std::uint64_t>& GetOutTensorsRanks() const = 0;

  virtual void Equal(const Iterator& lhs, const Iterator& rhs) = 0;

  virtual void Equal(const Index& lhs, const Index& rhs) = 0;

  virtual void Equal(const IteratorTuple& lhs, const IteratorTuple& rhs) = 0;

  virtual Iterator GetBroadcastedInputIterator(const Iterator& out_iterator,
                                               const DimExpr& dim) = 0;

  virtual Iterator GetConstantIterator(const Index& in_index, int constant) = 0;

  virtual const IteratorTuple& GetInIteratorTuple(
      std::size_t input_idx) const = 0;

  virtual const IteratorTuple& GetOutIteratorTuple(
      std::size_t output_idx) const = 0;

  virtual const Index& GetInIndex(std::size_t input_idx) const = 0;

  virtual const Index& GetOutIndex(std::size_t output_idx) const = 0;

  virtual const DimTuple& GetInDimTuple(std::size_t input_idx) const = 0;

  virtual const DimTuple& GetOutDimTuple(std::size_t output_idx) const = 0;

  template <typename T>
  const T& Attr(const std::string& name) const {
    return absl::get<T>(GetAttribute(name));
  }

 protected:
  OpEquationContext() = default;

  virtual const utils::Attribute& GetAttribute(
      const std::string& name) const = 0;
};

}  // namespace cinn::adt::config
