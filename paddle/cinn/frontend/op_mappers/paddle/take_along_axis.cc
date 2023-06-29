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
#include "paddle/cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void TakeAlongAxis2OpMapper(const paddle::cpp::OpDesc& op_desc,
                            const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("Input").size(), 1UL);
  auto x_name = op_desc.Input("Input").front();
  auto x = ctx.GetVar(x_name);
  CHECK_EQ(op_desc.Input("Index").size(), 1UL);
  auto index_name = op_desc.Input("Index").front();
  auto index = ctx.GetVar(index_name);

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "Axis");

  VLOG(4) << "x shape: " << cinn::utils::Join(x->shape, ",");
  VLOG(4) << "index shape: " << cinn::utils::Join(index->shape, ",");
  VLOG(4) << "take_along_axis axis: " << axis;

  auto out = ctx.Builder()->Gather(x, index, axis);

  CHECK_EQ(op_desc.Output("Result").size(), 1UL);
  auto out_name = op_desc.Output("Result").front();
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_take_along_axis) {
  CINN_REGISTER_OP_MAPPER(
      take_along_axis, cinn::frontend::paddle_mappers::TakeAlongAxis2OpMapper)
  return true;
}
