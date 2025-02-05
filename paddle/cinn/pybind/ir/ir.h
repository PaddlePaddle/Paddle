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

#include <string>
#include <vector>
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/pybind/ir/ir_context.h"
namespace cinn {
namespace pybind {

template IRContext IRBuilderNode::GetLastContext<ScheduleBlockContextNode>()
    const;
Var SetScheduleBlockIterVar(Var iter_var, Expr expr);
std::vector<Expr> AxisMap(const std::string &kinds,
                          const std::vector<Expr> &iter_expression);
void TensorStore(Expr tensor, Expr value, const std::vector<Expr> &indices);
Expr Arg(const std::string &name, Var var);
Expr Arg(const std::string &name, ir::Buffer buffer);
IRContext Sequential(Expr min, Expr extent);
}  // namespace pybind
}  // namespace cinn
