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

#include "paddle/cinn/adt/print_utils/print_schedule_dim.h"
#include "paddle/cinn/adt/print_utils/print_dim_expr.h"
#include "paddle/cinn/adt/schedule_dim.h"

namespace cinn::adt {

namespace {

std::string ToTxtStringScheduleDimImpl(const tReduced<LoopSize>& loop_size) {
  return "R(" + ToTxtString(loop_size.value()) + ")";
}

std::string ToTxtStringScheduleDimImpl(const tInjective<LoopSize>& loop_size) {
  return "I(" + ToTxtString(loop_size.value()) + ")";
}

}  // namespace

std::string ToTxtString(const ScheduleDim& schedule_dim) {
  return std::visit(
      [&](const auto& impl) { return ToTxtStringScheduleDimImpl(impl); },
      schedule_dim.variant());
}

std::string ToTxtString(const List<ScheduleDim>& schedule_dims) {
  std::string ret;
  ret += "[";
  int count = 0;
  for (const auto& schedule_dim : *schedule_dims) {
    if (count++ != 0) {
      ret += ", ";
    }
    ret += ToTxtString(schedule_dim);
  }
  ret += "]";
  return ret;
}

}  // namespace cinn::adt
