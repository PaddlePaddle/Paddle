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

#include "paddle/cinn/adt/get_sub_reshape_dim_ranges.h"

#include "paddle/cinn/adt/dim_expr.h"

namespace cinn::adt {

namespace {

std::int64_t GetNumel(const List<DimExpr>& constants) {
  std::int64_t ret = 1;
  for (const auto& constant : *constants) {
    ret *= constant.Get<std::int64_t>();
  }
  return ret;
}

}  // namespace

std::optional<std::tuple<std::vector<std::pair<int, int>>,
                         std::vector<std::pair<int, int>>>>
GetSubReshapeDimRanges(const List<DimExpr>& lhs_dims,
                       const List<DimExpr>& rhs_dims) {
  if (GetNumel(lhs_dims) != GetNumel(rhs_dims)) {
    return std::nullopt;
  }
  PADDLE_ENFORCE_EQ(
      !lhs_dims->empty(),
      true,
      ::common::errors::InvalidArgument("Sorry,but lhs_dims is empty"));
  PADDLE_ENFORCE_EQ(
      !rhs_dims->empty(),
      true,
      ::common::errors::InvalidArgument("Sory,but rhs_dims is empty"));
  std::vector<std::pair<int, int>> lhs_ranges{};
  std::vector<std::pair<int, int>> rhs_ranges{};
  int lhs_start = 0;
  int rhs_start = 0;
  int lhs_end = 0;
  int rhs_end = 0;

  const auto GetProduct = [&](const List<DimExpr>& dims,
                              std::size_t end) -> std::int64_t {
    end = (end > dims->size() ? dims->size() : end);
    std::int64_t ret = 1;
    for (std::size_t i = 0; i < end; ++i) {
      PADDLE_ENFORCE_EQ(
          dims->at(i).Has<std::int64_t>(),
          true,
          ::common::errors::InvalidArgument("dims->at(i) is not int64_t"));
      ret *= dims->at(i).Get<std::int64_t>();
    }
    return ret;
  };

  const auto LhsAcc = [&]() -> std::int64_t {
    return GetProduct(lhs_dims, lhs_end);
  };

  const auto RhsAcc = [&]() -> std::int64_t {
    return GetProduct(rhs_dims, rhs_end);
  };

  while (lhs_end < lhs_dims->size() || rhs_end < rhs_dims->size()) {
    if (lhs_start == lhs_end) {
      lhs_end++;
    }
    if (rhs_start == rhs_end) {
      rhs_end++;
    }
    if (LhsAcc() == RhsAcc()) {
      lhs_ranges.emplace_back(std::make_pair(lhs_start, lhs_end));
      rhs_ranges.emplace_back(std::make_pair(rhs_start, rhs_end));
      lhs_start = lhs_end;
      rhs_start = rhs_end;
    } else if (LhsAcc() < RhsAcc()) {
      lhs_end++;
    } else if (LhsAcc() > RhsAcc()) {
      rhs_end++;
    } else {
      PADDLE_THROW(::common::errors::Fatal("Dead code"));
    }
  }
  PADDLE_ENFORCE_EQ(lhs_end == lhs_dims->size() && rhs_end == rhs_dims->size(),
                    true,
                    ::common::errors::InvalidArgument(
                        "lhs_end is not equal to lhs_dims->size() and rhs_end "
                        "is not equal to rhs_dims->size()"));
  if (lhs_start < lhs_end && rhs_start < rhs_end) {
    lhs_ranges.emplace_back(std::make_pair(lhs_start, lhs_end));
    rhs_ranges.emplace_back(std::make_pair(rhs_start, rhs_end));
  }
  return std::make_tuple(lhs_ranges, rhs_ranges);
}

}  // namespace cinn::adt
