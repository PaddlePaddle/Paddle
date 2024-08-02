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
#include <glog/logging.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "paddle/common/enforce.h"
namespace cinn {
namespace ir {

struct Var;
struct Expr;

}  // namespace ir
}  // namespace cinn

namespace cinn {
namespace common {

//! Get the predefined axis name.
std::string axis_name(int level);

//! Generate `naxis` axis using the global names (i,j,k...).
std::vector<ir::Var> GenDefaultAxis(int naxis);
std::vector<ir::Expr> GenDefaultAxisAsExpr(int naxis);

bool IsAxisNameReserved(const std::string& x);

}  // namespace common
}  // namespace cinn
