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

#include <variant>

#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void Atan2OpMapper(const paddle::cpp::OpDesc& op_desc,
                   const cinn::frontend::OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X1").size(), 1UL);
  auto x1_name = op_desc.Input("X1").front();
  CHECK_EQ(op_desc.Input("X2").size(), 1UL);
  auto x2_name = op_desc.Input("X2").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x1 = ctx.GetVar(x1_name);
  auto x2 = ctx.GetVar(x2_name);

  if (x1->type.is_int() && x2->type.is_int()) {
    x1 = ctx.Builder()->Cast(x1, "float64");
    x2 = ctx.Builder()->Cast(x2, "float64");
  }

  auto out = ctx.Builder()->Atan2(x1, x2);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_atan) {
  CINN_REGISTER_OP_MAPPER(atan2, cinn::frontend::paddle_mappers::Atan2OpMapper)
  return true;
}
