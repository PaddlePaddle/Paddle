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
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/hlir/framework/node.h"

namespace cinn::adt::config {

class ConditionalEqualHandler {
 public:
  ConditionalEqualHandler(const ConditionalEqualHandler&) = delete;
  ConditionalEqualHandler(ConditionalEqualHandler&&) = delete;
  virtual ~ConditionalEqualHandler() = default;

  virtual void Where(const EquationStaticLogical&) const = 0;

 protected:
  ConditionalEqualHandler() = default;
};

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

  virtual std::
      unique_ptr<ConditionalEqualHandler> [[nodiscard]] ConditionalEqual(
          const Iterator& lhs, const Iterator& rhs) = 0;

  virtual std::
      unique_ptr<ConditionalEqualHandler> [[nodiscard]] ConditionalEqual(
          const Iterator& iterator, std::size_t constant) = 0;

  virtual std::unique_ptr<
      ConditionalEqualHandler> [[nodiscard]] ConditionalEqual(const Index& lhs,
                                                              const Index&
                                                                  rhs) = 0;

  virtual EquationStaticLogical EQ(const Dim& lhs, const Dim& rhs) const = 0;

  virtual EquationStaticLogical NE(const Dim& lhs, const Dim& rhs) const = 0;

  virtual const IteratorTuple& GetInIteratorTuple(
      std::size_t input_idx) const = 0;

  virtual const IteratorTuple& GetOutIteratorTuple(
      std::size_t output_idx) const = 0;

  virtual const Index& GetInIndex(std::size_t input_idx) const = 0;

  virtual const Index& GetOutIndex(std::size_t output_idx) const = 0;

  virtual const StrideTuple& GetInStrideTuple(std::size_t input_idx) const = 0;

  virtual const StrideTuple& GetOutStrideTuple(
      std::size_t output_idx) const = 0;

  virtual const DimTuple& GetInDimTuple(std::size_t input_idx) const = 0;

  virtual const DimTuple& GetOutDimTuple(std::size_t output_idx) const = 0;

 protected:
  OpEquationContext() = default;
};

}  // namespace cinn::adt::config
