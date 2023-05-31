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

#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void DropoutInferOpMapper(const paddle::cpp::OpDesc& op_desc,
                          const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto dropout_prob =
      utils::GetAttrOrDefault<float>(op_desc, "dropout_prob", 0.5f);
  auto dropout_implementation = utils::GetAttrOrDefault<std::string>(
      op_desc, "dropout_implementation", "downgrade_in_infer");
  auto x = ctx.GetVar(x_name);
  auto out =
      ctx.Builder()->DropoutInfer(x, dropout_prob, dropout_implementation);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_dropout) {
  CINN_REGISTER_OP_MAPPER(dropout,
                          cinn::frontend::paddle_mappers::DropoutInferOpMapper)
  return true;
}
