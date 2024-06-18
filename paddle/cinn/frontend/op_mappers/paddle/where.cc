// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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
#include "paddle/common/enforce.h"
namespace cinn {
namespace frontend {
namespace paddle_mappers {

void WhereOpMapper(const paddle::cpp::OpDesc& op_desc,
                   const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("Condition").size(),
      1UL,
      phi::errors::InvalidArgument("The input of where op must be 1"));
  auto c_name = op_desc.Input("Condition").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of where op must be 1"));
  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Input("Y").size(),
      1UL,
      phi::errors::InvalidArgument("The input of where op must be 1"));
  auto y_name = op_desc.Input("Y").front();

  PADDLE_ENFORCE_EQ(
      op_desc.Output("Out").size(),
      1UL,
      phi::errors::InvalidArgument("The output of where op must be 1"));
  auto out_name = op_desc.Output("Out").front();

  auto c = ctx.GetVar(c_name);
  auto x = ctx.GetVar(x_name);
  auto y = ctx.GetVar(y_name);

  auto out = ctx.Builder()->Select(c, x, y);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_where) {
  CINN_REGISTER_OP_MAPPER(where, cinn::frontend::paddle_mappers::WhereOpMapper)
  return true;
}
