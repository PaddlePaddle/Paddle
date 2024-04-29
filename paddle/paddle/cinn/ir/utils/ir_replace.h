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
#include <string>

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace ir {
namespace ir_utils {

//! Replace the variable \p from to expression \p to in expression \p expr.
void IrReplaceVarBroadcast(ir::Expr* expr, ir::Expr from, ir::Expr to);

//! Replace the Expr \p from to expression \p to in expression \p expr.
void IrReplace(ir::Expr* expr, ir::Expr from, ir::Expr to);

}  // namespace ir_utils
}  // namespace ir
}  // namespace cinn
