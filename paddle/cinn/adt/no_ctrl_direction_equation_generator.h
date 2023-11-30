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

#include "paddle/cinn/adt/direction_equation_generator.h"

namespace cinn::adt {

class NoCtrlDirectionEquationGenerator final
    : public DirectionEquationGenerator {
 public:
  NoCtrlDirectionEquationGenerator(const NoCtrlDirectionEquationGenerator&) =
      delete;
  NoCtrlDirectionEquationGenerator(NoCtrlDirectionEquationGenerator&&) = delete;

  NoCtrlDirectionEquationGenerator();

  Equations GetDirectionEquations() const override;

  std::function<const OpStmt*(const FakeOpPlaceHolder&)>
  MakeGetterOpStmt4OpPlaceHolder() const override;

  std::optional<Index> OutMsgIndex4InMsgIndex(
      const Index& index) const override;

 private:
};

}  // namespace cinn::adt
