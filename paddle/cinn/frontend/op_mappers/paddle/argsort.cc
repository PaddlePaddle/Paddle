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

#include <variant>

#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void ArgsortOpMapper(const paddle::cpp::OpDesc& op_desc,
                     const cinn::frontend::OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();

  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  CHECK_EQ(op_desc.Output("Indices").size(), 1UL);
  auto indices_name = op_desc.Output("Indices").front();

  auto is_ascend =
      !(utils::GetAttrOrDefault<bool>(op_desc, "descending", false));
  auto axis =
      utils::ToDimType(utils::GetAttrOrDefault<int64_t>(op_desc, "axis", 0));

  auto x = ctx.GetVar(x_name);
  auto out = ctx.Builder()->ArgSort(x, axis, is_ascend);
  auto idx = ctx.Builder()->Cast(out[0], "int64");

  ctx.AddVar(indices_name, idx);
  ctx.AddVarModelToProgram(indices_name, idx->id);

  // TODO(lanxianghit): return the sorted tensor here. Now out[1] is a temporary
  // tensor. this is because output 'Out' is never uesd in Paddle API, but CINN
  // need to return 2 output vars to meet the op defination, this should be
  // resolved after sort op restructured.
  ctx.AddVar(out_name, out[1]);
  ctx.AddVarModelToProgram(out_name, out[1]->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_argsort) {
  CINN_REGISTER_OP_MAPPER(argsort,
                          cinn::frontend::paddle_mappers::ArgsortOpMapper)
  return true;
}
