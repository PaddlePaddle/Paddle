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
#include <string>

#include "paddle/cinn/ir/ir.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace optim {

/**
 * Given Expr AST, analyze the Buffer size by loop var range.
 * Then resize the buffer to the range. It is an optimization
 * so that we can allocate buffer to the size accessed by vars,
 * which minimize the allocation.
 */
void ResizeBufferToMaxVarRange(ir::Expr* expr);

}  // namespace optim
}  // namespace cinn
