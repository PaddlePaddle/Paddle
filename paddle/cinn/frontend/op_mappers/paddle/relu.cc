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

void ReluOpMapper(const paddle::cpp::OpDesc& op_desc,
                  const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  auto x = ctx.GetVar(x_name);
  auto out = ctx.Builder()->Relu(x);

  ctx.AddVar(out_name, out, true);
  ctx.AddVarModelToProgram(out_name, out->id, true);
}

void Relu6OpMapper(const paddle::cpp::OpDesc& op_desc,
                   const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto threshold = utils::GetAttrOrDefault<float>(op_desc, "threshold", 6.0f);
  auto x = ctx.GetVar(x_name);
  auto out = ctx.Builder()->Relu6(x, threshold);

  ctx.AddVar(out_name, out, true);
  ctx.AddVarModelToProgram(out_name, out->id, true);
}

void ReluGradOpMapper(const paddle::cpp::OpDesc& op_desc,
                      const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input(paddle::GradVarName("Out")).size(), 1UL);
  auto dout_name = op_desc.Input(paddle::GradVarName("Out")).front();
  CHECK_EQ(op_desc.Input("Out").size(), 1UL);
  auto out_name = op_desc.Input("Out").front();
  CHECK_EQ(op_desc.Output(paddle::GradVarName("X")).size(), 1UL);
  auto dx_name = op_desc.Output(paddle::GradVarName("X")).front();

  auto dout = ctx.GetVar(dout_name);
  auto out = ctx.GetVar(out_name);
  auto dx = ctx.Builder()->ReluGrad(dout, out);

  ctx.AddVar(dx_name, dx, true);
  ctx.AddVarModelToProgram(dx_name, dx->id, true);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_relu) {
  CINN_REGISTER_OP_MAPPER(relu, cinn::frontend::paddle_mappers::ReluOpMapper)
  CINN_REGISTER_OP_MAPPER(relu_grad,
                          cinn::frontend::paddle_mappers::ReluGradOpMapper)
  CINN_REGISTER_OP_MAPPER(relu6, cinn::frontend::paddle_mappers::Relu6OpMapper)
  return true;
}
