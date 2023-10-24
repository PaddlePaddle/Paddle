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
namespace science_mappers {

#define BINARY_OPMAPPER(op_name)                                      \
  void op_name##OpMapper(const paddle::cpp::OpDesc& op_desc,          \
                         const OpMapperContext& ctx) {                \
    CHECK_EQ(op_desc.Input("X").size(), 1UL);                         \
    auto x_name = op_desc.Input("X").front();                         \
    CHECK_EQ(op_desc.Input("Y").size(), 1UL);                         \
    auto y_name = op_desc.Input("Y").front();                         \
    CHECK_EQ(op_desc.Output("Z").size(), 1UL);                        \
    auto out_name = op_desc.Output("Z").front();                      \
    VLOG(3) << out_name << " = " << #op_name << "(" << x_name << ", " \
            << y_name << ")";                                         \
    auto x = ctx.GetVar(x_name);                                      \
    auto y = ctx.GetVar(y_name);                                      \
    auto out = ctx.Builder()->op_name(x, y);                          \
    ctx.AddVar(out_name, out);                                        \
    ctx.AddVarModelToProgram(out_name, out->id);                      \
  }

BINARY_OPMAPPER(Add)
BINARY_OPMAPPER(Subtract)
BINARY_OPMAPPER(Divide)
BINARY_OPMAPPER(Multiply)
BINARY_OPMAPPER(Matmul)
BINARY_OPMAPPER(Pow)
BINARY_OPMAPPER(Max)
BINARY_OPMAPPER(Min)

#undef BINARY_OPMAPPER

#define UNARY_OPMAPPER(op_name)                                       \
  void op_name##OpMapper(const paddle::cpp::OpDesc& op_desc,          \
                         const OpMapperContext& ctx) {                \
    CHECK_EQ(op_desc.Input("X").size(), 1UL);                         \
    auto x_name = op_desc.Input("X").front();                         \
    CHECK_EQ(op_desc.Output("Y").size(), 1UL);                        \
    auto out_name = op_desc.Output("Y").front();                      \
    VLOG(3) << out_name << " = " << #op_name << "(" << x_name << ")"; \
    auto x = ctx.GetVar(x_name);                                      \
    auto out = ctx.Builder()->op_name(x);                             \
    ctx.AddVar(out_name, out);                                        \
    ctx.AddVarModelToProgram(out_name, out->id);                      \
  }

UNARY_OPMAPPER(Sqrt)
UNARY_OPMAPPER(Rsqrt)
UNARY_OPMAPPER(Tanh)
UNARY_OPMAPPER(Sin)
UNARY_OPMAPPER(Cos)
UNARY_OPMAPPER(Exp)
UNARY_OPMAPPER(Erf)
UNARY_OPMAPPER(Log)
UNARY_OPMAPPER(Identity)
UNARY_OPMAPPER(Abs)

#undef UNARY_OPMAPPER

}  // namespace science_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(science_math) {
#define EXPAND_OP_MAPPER_REGISTER(psci_op, cinn_op) \
  CINN_REGISTER_OP_MAPPER(psci_op,                  \
                          cinn::frontend::science_mappers::cinn_op##OpMapper)

  EXPAND_OP_MAPPER_REGISTER(add_p, Add)
  EXPAND_OP_MAPPER_REGISTER(sub_p, Subtract)
  EXPAND_OP_MAPPER_REGISTER(div_p, Divide)
  EXPAND_OP_MAPPER_REGISTER(mul_p, Multiply)
  EXPAND_OP_MAPPER_REGISTER(matmul_p, Matmul)
  EXPAND_OP_MAPPER_REGISTER(pow_p, Pow)
  EXPAND_OP_MAPPER_REGISTER(max_p, Max)
  EXPAND_OP_MAPPER_REGISTER(min_p, Min)

  EXPAND_OP_MAPPER_REGISTER(sqrt_p, Sqrt)
  EXPAND_OP_MAPPER_REGISTER(rsqrt_p, Rsqrt)
  EXPAND_OP_MAPPER_REGISTER(tanh_p, Tanh)
  EXPAND_OP_MAPPER_REGISTER(sin_p, Sin)
  EXPAND_OP_MAPPER_REGISTER(cos_p, Cos)
  EXPAND_OP_MAPPER_REGISTER(exp_p, Exp)
  EXPAND_OP_MAPPER_REGISTER(erf_p, Erf)
  EXPAND_OP_MAPPER_REGISTER(log_p, Log)
  EXPAND_OP_MAPPER_REGISTER(clone_p, Identity)
  EXPAND_OP_MAPPER_REGISTER(share_data_p, Identity)
  EXPAND_OP_MAPPER_REGISTER(abs_p, Abs)

#undef EXPAND_OP_MAPPER_REGISTER

  return true;
}
