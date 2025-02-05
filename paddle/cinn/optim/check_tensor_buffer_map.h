// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/intrinsic_ops.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/tensor.h"

namespace cinn {
namespace optim {
/*
 * Debugger for detecting whether a tensor name maps multiple buffers
 */
void CheckTensorBufferMap(const std::vector<ir::Expr> &expr,
                          const std::string &process);

void CheckTensorBufferMap(const std::vector<ir::Expr *> &expr,
                          const std::string &process);

void CheckTensorBufferMap(const Expr *expr, const std::string &process);

void CheckTensorBufferMap(const Expr &expr, const std::string &process);

bool CheckTensorBufferMap(const Expr *expr);

bool CheckTensorBufferMap(const Expr &expr);

}  // namespace optim
}  // namespace cinn
