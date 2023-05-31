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

#define UNARY_OPMAPPER_FUNCTION(OP_NAME)                                    \
  void OP_NAME##OpMapper(const paddle::cpp::OpDesc& op_desc,                \
                         const OpMapperContext& ctx) {                      \
    CHECK_EQ(op_desc.Input("X").size(), 1UL);                               \
    auto x_name = op_desc.Input("X").front();                               \
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);                            \
    auto out_name = op_desc.Output("Out").front();                          \
    auto x = ctx.GetVar(x_name);                                            \
    VLOG(4) << #OP_NAME << " X:" << x_name << "["                           \
            << cinn::utils::Join(x->shape, ",") << "] to Out:" << out_name; \
    auto out = ctx.Builder()->OP_NAME(x);                                   \
    ctx.AddVar(out_name, out);                                              \
    ctx.AddVarModelToProgram(out_name, out->id);                            \
  }

UNARY_OPMAPPER_FUNCTION(LogicalNot)
UNARY_OPMAPPER_FUNCTION(BitwiseNot)
UNARY_OPMAPPER_FUNCTION(Sqrt)
UNARY_OPMAPPER_FUNCTION(Gelu)
UNARY_OPMAPPER_FUNCTION(Sigmoid)
UNARY_OPMAPPER_FUNCTION(Exp)
UNARY_OPMAPPER_FUNCTION(Erf)
UNARY_OPMAPPER_FUNCTION(Rsqrt)
UNARY_OPMAPPER_FUNCTION(Floor)
UNARY_OPMAPPER_FUNCTION(Ceil)
UNARY_OPMAPPER_FUNCTION(Round)
UNARY_OPMAPPER_FUNCTION(Trunc)
UNARY_OPMAPPER_FUNCTION(Sin)
UNARY_OPMAPPER_FUNCTION(Cos)
UNARY_OPMAPPER_FUNCTION(Tan)
UNARY_OPMAPPER_FUNCTION(Sinh)
UNARY_OPMAPPER_FUNCTION(Cosh)
UNARY_OPMAPPER_FUNCTION(Tanh)
UNARY_OPMAPPER_FUNCTION(Asin)
UNARY_OPMAPPER_FUNCTION(Acos)
UNARY_OPMAPPER_FUNCTION(Atan)
UNARY_OPMAPPER_FUNCTION(Asinh)
UNARY_OPMAPPER_FUNCTION(Acosh)
UNARY_OPMAPPER_FUNCTION(Atanh)
UNARY_OPMAPPER_FUNCTION(Sign)
UNARY_OPMAPPER_FUNCTION(Abs)
UNARY_OPMAPPER_FUNCTION(Reciprocal)
UNARY_OPMAPPER_FUNCTION(IsNan)
UNARY_OPMAPPER_FUNCTION(IsFinite)
UNARY_OPMAPPER_FUNCTION(IsInf)

#undef UNARY_OPMAPPER_FUNCTION
}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_unary) {
#define UNARY_OPMAPPER_REGISTER(PD_OP, CINN_OP) \
  CINN_REGISTER_OP_MAPPER(PD_OP,                \
                          cinn::frontend::paddle_mappers::CINN_OP##OpMapper)

  UNARY_OPMAPPER_REGISTER(logical_not, LogicalNot)
  UNARY_OPMAPPER_REGISTER(bitwise_not, BitwiseNot)
  UNARY_OPMAPPER_REGISTER(sqrt, Sqrt)
  UNARY_OPMAPPER_REGISTER(gelu, Gelu)
  UNARY_OPMAPPER_REGISTER(sigmoid, Sigmoid)
  UNARY_OPMAPPER_REGISTER(exp, Exp)
  UNARY_OPMAPPER_REGISTER(erf, Erf)
  UNARY_OPMAPPER_REGISTER(rsqrt, Rsqrt)
  UNARY_OPMAPPER_REGISTER(floor, Floor)
  UNARY_OPMAPPER_REGISTER(ceil, Ceil)
  UNARY_OPMAPPER_REGISTER(round, Round)
  UNARY_OPMAPPER_REGISTER(trunc, Trunc)
  UNARY_OPMAPPER_REGISTER(sin, Sin)
  UNARY_OPMAPPER_REGISTER(cos, Cos)
  UNARY_OPMAPPER_REGISTER(tan, Tan)
  UNARY_OPMAPPER_REGISTER(sinh, Sinh)
  UNARY_OPMAPPER_REGISTER(cosh, Cosh)
  UNARY_OPMAPPER_REGISTER(tanh, Tanh)
  UNARY_OPMAPPER_REGISTER(asin, Asin)
  UNARY_OPMAPPER_REGISTER(acos, Acos)
  UNARY_OPMAPPER_REGISTER(atan, Atan)
  UNARY_OPMAPPER_REGISTER(asinh, Asinh)
  UNARY_OPMAPPER_REGISTER(acosh, Acosh)
  UNARY_OPMAPPER_REGISTER(atanh, Atanh)
  UNARY_OPMAPPER_REGISTER(sign, Sign)
  UNARY_OPMAPPER_REGISTER(abs, Abs)
  UNARY_OPMAPPER_REGISTER(reciprocal, Reciprocal)
  UNARY_OPMAPPER_REGISTER(isinf_v2, IsInf)
  UNARY_OPMAPPER_REGISTER(isnan_v2, IsNan)
  UNARY_OPMAPPER_REGISTER(isfinite_v2, IsFinite)

#undef UNARY_OPMAPPER_REGISTER

  return true;
}
