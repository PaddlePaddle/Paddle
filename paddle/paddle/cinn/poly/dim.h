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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/ir/ir_base.h"

/**
 * \file
 * This file defines Dim class, which represents the dimension in polyhedral.
 */

namespace cinn {
namespace poly {

/**
 * Dimension with name and range.
 *
 * This is used in ISL to define each dimension of a statement.
 */
struct Dim {
  using value_t = ir::Expr;
  using range_t = std::pair<value_t, value_t>;

  //! The id of the dimension.
  std::string id;
  //! The lower bound.
  value_t lower_bound;
  //! The upper bound.
  value_t upper_bound;

  //! Construct a parameter.
  explicit Dim(std::string id) : id(std::move(id)) {}

  //! Construct a dimension with integer range.
  Dim(std::string id, uint32_t lower_bound, uint32_t upper_bound)
      : id(std::move(id)), lower_bound(lower_bound), upper_bound(upper_bound) {}

  //! Construct a dimension with int64_t range.
  Dim(std::string id, int64_t lower_bound, int64_t upper_bound)
      : id(std::move(id)), lower_bound(lower_bound), upper_bound(upper_bound) {}

  //! Construct a dimension with expression range.
  Dim(std::string id, ir::Expr lower_bound, ir::Expr upper_bound);

  //! Return the range composed of (lower_bound, upper_bound).
  range_t range() const { return std::make_pair(lower_bound, upper_bound); }

  bool is_param() const {
    return !lower_bound.defined() && !lower_bound.defined();
  }

  //! Return the ISL style range representation, such as '0 <= i <= 20'.
  std::string range_repr() const;
};

}  // namespace poly
}  // namespace cinn
