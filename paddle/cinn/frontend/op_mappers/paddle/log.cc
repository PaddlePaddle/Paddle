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

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void LogOpMapper(const paddle::cpp::OpDesc& op_desc,
                 const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x = ctx.GetVar(x_name);
  auto out = ctx.Builder()->Log(x);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void Log2OpMapper(const paddle::cpp::OpDesc& op_desc,
                  const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x = ctx.GetVar(x_name);
  auto out = ctx.Builder()->Log2(x);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void Log10OpMapper(const paddle::cpp::OpDesc& op_desc,
                   const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x = ctx.GetVar(x_name);
  auto out = ctx.Builder()->Log10(x);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void Log1pOpMapper(const paddle::cpp::OpDesc& op_desc,
                   const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto x = ctx.GetVar(x_name);

  auto one = ctx.Builder()->FillConstant(x->shape,
                                         1.0f,
                                         cinn::UniqName(x->id + "_1p"),
                                         cinn::common::Type2Str(x->type));
  auto y = ctx.Builder()->Add(x, one);
  auto out = ctx.Builder()->Log(y);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_log) {
  CINN_REGISTER_OP_MAPPER(log, cinn::frontend::paddle_mappers::LogOpMapper)
  CINN_REGISTER_OP_MAPPER(log2, cinn::frontend::paddle_mappers::Log2OpMapper)
  CINN_REGISTER_OP_MAPPER(log10, cinn::frontend::paddle_mappers::Log10OpMapper)
  CINN_REGISTER_OP_MAPPER(log1p, cinn::frontend::paddle_mappers::Log1pOpMapper)
  return true;
}
