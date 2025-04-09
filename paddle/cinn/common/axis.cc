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

#include "paddle/cinn/common/axis.h"

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/poly/dim.h"
#include "paddle/cinn/poly/domain.h"
#include "paddle/cinn/poly/stage.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace common {

static const std::vector<std::string> kAxes({
    "i",  // level 0
    "j",  // level 1
    "k",  // level 2
    "a",  // level 3
    "b",  // level 4
    "c",  // level 5
    "d",  // level 6
    "e",  // level 7
    "f",  // level 8
    "g",  // level 9
    "h",  // level 10
    "l",  // level 11
    "m",  // level 12
    "n",  // level 13
    "o",  // level 14
    "p",  // level 15
    "q",  // level 16
    "r",  // level 17
    "s",  // level 18
    "t",  // level 19
    "u",  // level 20
    "v"   // level 21
});

std::string axis_name(int level) {
  if (level < kAxes.size()) {
    return kAxes[level];
  }
  // upper level
  int repeat_num = 1 + (level / kAxes.size());
  const auto& base_axis = kAxes[level % kAxes.size()];

  // if the level greater than kAxis, repeat the axis, like:
  // level == 22 ==> axis = "ii"
  std::string axis;
  for (int i = 0; i < repeat_num; ++i) {
    axis.append(base_axis);
  }
  return axis;
}

std::vector<ir::Var> GenDefaultAxis(int naxis) {
  std::vector<ir::Var> axis;
  for (int i = 0; i < naxis; i++) {
    axis.emplace_back(cinn::common::axis_name(i));
    PADDLE_ENFORCE_EQ(axis.back()->type().valid(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The type of axis at index %d is invalid. Please "
                          "ensure each axis has a valid type.",
                          i));
  }
  return axis;
}

std::vector<ir::Expr> GenDefaultAxisAsExpr(int naxis) {
  auto vars = GenDefaultAxis(naxis);
  std::vector<Expr> res;
  for (auto& v : vars) {
    res.push_back(Expr(v));
  }
  return res;
}

static const std::set<std::string>& axis_set() {
  static std::set<std::string> x(kAxes.begin(), kAxes.end());
  return x;
}

bool IsAxisNameReserved(const std::string& x) {
  if (x.empty()) {
    // axis should not be empty
    return false;
  }
  if (axis_set().count(x)) {
    return true;
  }
  if (!axis_set().count(std::string(1, x[0]))) {
    // all char in axis should in kAxes
    return false;
  }
  bool is_repeat_axis = true;
  for (int i = 1; i < x.size(); ++i) {
    if (x[i] != x[0]) {
      // the axis are repeat with the char in kAxes
      is_repeat_axis = false;
      break;
    }
  }
  return is_repeat_axis;
}

}  // namespace common
}  // namespace cinn
