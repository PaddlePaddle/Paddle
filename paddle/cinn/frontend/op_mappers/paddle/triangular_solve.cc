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

#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void TriangularSolveOpMapper(const paddle::cpp::OpDesc& op_desc,
                             const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  constexpr bool left_side = true;
  auto upper = utils::GetAttrOrDefault<bool>(op_desc, "upper", true);
  auto transpose = utils::GetAttrOrDefault<bool>(op_desc, "transpose", false);
  auto unitriangular =
      utils::GetAttrOrDefault<bool>(op_desc, "unitriangular", false);
  VLOG(4) << "out_name = triangular_solve(" << x_name
          << ", left_side=" << left_side << ", upper=" << upper
          << ", transpose=" << transpose << ", unitriangular=" << unitriangular
          << ")";

  auto x = ctx.GetVar(x_name);
  auto y = ctx.GetVar(y_name);
  auto out = ctx.Builder()->TriangularSolve(
      x, y, left_side, upper, transpose, unitriangular);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_triangular_solve) {
  CINN_REGISTER_OP_MAPPER(
      triangular_solve, cinn::frontend::paddle_mappers::TriangularSolveOpMapper)
  return true;
}
