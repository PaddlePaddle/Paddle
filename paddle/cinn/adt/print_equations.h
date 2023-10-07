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

#include <vector>
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/m_expr.h"
#include "paddle/cinn/adt/partition_op_stmts.h"

namespace cinn::adt {

std::string ToTxtString(const Equation& equation);

std::string ToTxtString(const Equations& equations,
                        const std::string& separator);

std::string ToTxtString(const Iterator& iterator);

std::string ToTxtString(const Index& index);

std::string ToTxtString(const List<Index>& index_list);

std::string ToTxtString(const FakeOpPlaceHolder& op);

std::string ToTxtString(const List<Iterator>& iterator_list);

std::string ToTxtString(const tInMsg<List<Index>>& in_msg_indexes_);

std::string ToTxtString(const tOutMsg<List<Index>>& out_msg_indexes_);

void PrintEquations(const Equations& equations, const std::string& separator);

void PrintOpStmtsEquations(const List<OpStmt>& op_stmts,
                           const EquationCtx4OpStmtT& EquationCtx4OpStmt);

void PrintIndexVector(const std::vector<Index>& indexes);

}  // namespace cinn::adt
