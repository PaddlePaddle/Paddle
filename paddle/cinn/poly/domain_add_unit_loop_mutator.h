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

#include <tuple>
#include <vector>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"

namespace cinn {
namespace poly {

/**
 * CINN Expr mutator utility to add length-1-loop to Expr based on input
 * dim names and dim range.
 */
class DomainAddUnitLoopMutator : public ir::IRMutator<> {
 public:
  DomainAddUnitLoopMutator(
      const std::vector<std::string>& dim_names,
      const std::vector<std::tuple<int, int, int>>& dim_min_max);

  void operator()(ir::Expr* expr);

 private:
  void Visit(const ir::For* op, Expr* expr) override;
  void Visit(const ir::PolyFor* op, Expr* expr) override;

  void MutateAfterVisit(ir::Expr* expr);

  std::vector<ir::For*> parent_for_;
  std::vector<ir::PolyFor*> parent_poly_for_;

  std::vector<ir::Expr> longest_loop_;

  const std::vector<std::string>& dim_names_;
  const std::vector<std::tuple<int, int, int>>& dim_min_max_;
};

}  // namespace poly
}  // namespace cinn
