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

#include <optional>

#include "paddle/cinn/adt/equation_function.h"

namespace cinn::adt {

class OpStmt;

class DirectionEquationGenerator {
 public:
  DirectionEquationGenerator(const DirectionEquationGenerator&) = delete;
  DirectionEquationGenerator(DirectionEquationGenerator&&) = delete;
  ~DirectionEquationGenerator() = default;

  virtual Equations GetDirectionEquations() const = 0;

  virtual std::function<const OpStmt*(const FakeOpPlaceHolder&)>
  MakeGetterOpStmt4OpPlaceHolder() const = 0;

  virtual std::optional<Index> OutMsgIndex4InMsgIndex(
      const Index& index) const = 0;

 protected:
  DirectionEquationGenerator() = default;
};

}  // namespace cinn::adt
