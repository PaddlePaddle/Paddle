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

void TopKOpMapper(const paddle::cpp::OpDesc& op_desc,
                  const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  CHECK_EQ(op_desc.Output("Indices").size(), 1UL);
  auto indices_name = op_desc.Output("Indices").front();
  auto x = ctx.GetVar(x_name);

  CHECK(op_desc.HasAttr("k"));
  auto k = utils::GetAttrOrDefault<int>(op_desc, "k");
  auto outs = ctx.Builder()->TopK(x, k, -1, true);

  ctx.AddVar(out_name, outs[0]);
  ctx.AddVarModelToProgram(out_name, outs[0]->id);
  ctx.AddVar(indices_name, outs[1]);
  ctx.AddVarModelToProgram(indices_name, outs[1]->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_top_k) {
  CINN_REGISTER_OP_MAPPER(top_k, cinn::frontend::paddle_mappers::TopKOpMapper)
  return true;
}
