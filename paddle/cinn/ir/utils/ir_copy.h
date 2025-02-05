// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <utility>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/lowered_func.h"

namespace cinn {

namespace ir {
class ModuleExpr;

namespace ir_utils {

//! Shallow copy an expression.
Expr IRCopy(const Expr& x, bool copy_buffer_node = true);

std::vector<Expr> IRCopy(const std::vector<Expr>& x,
                         bool copy_buffer_node = true);

ir::ModuleExpr IRCopy(const ir::ModuleExpr& x, bool copy_buffer_node = true);

ir::Module IRCopy(const Module& m, bool copy_buffer_node = true);

ir::LoweredFunc IRCopy(const ir::LoweredFunc& x, bool copy_buffer_node = true);

std::vector<ir::LoweredFunc> IRCopy(const std::vector<ir::LoweredFunc>& x,
                                    bool copy_buffer_node = true);

}  // namespace ir_utils
}  // namespace ir
}  // namespace cinn
