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

#include "paddle/cinn/adt/print_utils/print_dim_expr.h"

namespace cinn::adt {

std::string ToTxtString(const DimExpr& dim_expr) { return ToString(dim_expr); }

std::string ToTxtString(const List<DimExpr>& dim_exprs) {
  std::string ret;
  ret += "[";
  for (std::size_t idx = 0; idx < dim_exprs->size(); ++idx) {
    if (idx != 0) {
      ret += ", ";
    }
    ret += ToString(dim_exprs.Get(idx));
  }
  ret += "]";
  return ret;
}

}  // namespace cinn::adt
