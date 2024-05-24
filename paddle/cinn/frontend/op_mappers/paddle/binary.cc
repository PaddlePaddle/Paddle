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

#define BINARY_OPMAPPER_FUNCTION(OP_NAME)                             \
  void OP_NAME##OpMapper(const paddle::cpp::OpDesc& op_desc,          \
                         const OpMapperContext& ctx) {                \
    PADDLE_ENFORCE_EQ(                                                \
        op_desc.Input("X").size(),                                    \
        1UL,                                                          \
        phi::errors::InvalidArgument("The input of op must be 1."));  \
    auto x_name = op_desc.Input("X").front();                         \
    PADDLE_ENFORCE_EQ(                                                \
        op_desc.Input("Y").size(),                                    \
        1UL,                                                          \
        phi::errors::InvalidArgument("The input of op must be 1."));  \
    auto y_name = op_desc.Input("Y").front();                         \
    PADDLE_ENFORCE_EQ(                                                \
        op_desc.Output("Out").size(),                                 \
        1UL,                                                          \
        phi::errors::InvalidArgument("The output of op must be 1.")); \
    auto out_name = op_desc.Output("Out").front();                    \
    auto x = ctx.GetVar(x_name);                                      \
    auto y = ctx.GetVar(y_name);                                      \
    auto out = ctx.Builder()->OP_NAME(x, y);                          \
    ctx.AddVar(out_name, out);                                        \
    ctx.AddVarModelToProgram(out_name, out->id);                      \
  }

BINARY_OPMAPPER_FUNCTION(LogicalAnd)
BINARY_OPMAPPER_FUNCTION(LogicalOr)
BINARY_OPMAPPER_FUNCTION(LogicalXor)
BINARY_OPMAPPER_FUNCTION(BitwiseAnd)
BINARY_OPMAPPER_FUNCTION(BitwiseOr)
BINARY_OPMAPPER_FUNCTION(BitwiseXor)

#undef BINARY_OPMAPPER_FUNCTION

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_binary) {
#define BINARY_OPMAPPER_REGISTER(PD_OP, CINN_OP) \
  CINN_REGISTER_OP_MAPPER(PD_OP,                 \
                          cinn::frontend::paddle_mappers::CINN_OP##OpMapper)

  BINARY_OPMAPPER_REGISTER(logical_and, LogicalAnd)
  BINARY_OPMAPPER_REGISTER(logical_or, LogicalOr)
  BINARY_OPMAPPER_REGISTER(logical_xor, LogicalXor)
  BINARY_OPMAPPER_REGISTER(bitwise_and, BitwiseAnd)
  BINARY_OPMAPPER_REGISTER(bitwise_or, BitwiseOr)
  BINARY_OPMAPPER_REGISTER(bitwise_xor, BitwiseXor)

#undef BINARY_OPMAPPER_REGISTER

  return true;
}
