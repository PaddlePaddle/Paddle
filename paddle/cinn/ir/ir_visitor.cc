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

#include <unordered_set>

#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/utils/ir_compare.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace ir {

bool operator==(Expr a, Expr b) {
  if (a.get() == b.get()) return true;
  return ir_utils::IRCompare(a, b);
}

bool operator!=(Expr a, Expr b) { return !(a == b); }

}  // namespace ir
}  // namespace cinn
