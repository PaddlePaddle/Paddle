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
namespace science_mappers {

#define COMPARE_OPMAPPER_FUNCTION(OP_NAME)                             \
  void OP_NAME##OpMapper(const paddle::cpp::OpDesc& op_desc,           \
                         const OpMapperContext& ctx) {                 \
    PADDLE_ENFORCE_EQ(                                                 \
        op_desc.Input("X").size(),                                     \
        1UL,                                                           \
        phi::errors::InvalidArgument("The input of x op must be 1"));  \
    auto x_name = op_desc.Input("X").front();                          \
    PADDLE_ENFORCE_EQ(                                                 \
        op_desc.Input("Y").size(),                                     \
        1UL,                                                           \
        phi::errors::InvalidArgument("The input of y op must be 1"));  \
    auto y_name = op_desc.Input("Y").front();                          \
    PADDLE_ENFORCE_EQ(                                                 \
        op_desc.Output("Z").size(),                                    \
        1UL,                                                           \
        phi::errors::InvalidArgument("The output of z op must be 1")); \
    auto out_name = op_desc.Output("Z").front();                       \
    auto x = ctx.GetVar(x_name);                                       \
    auto y = ctx.GetVar(y_name);                                       \
    auto out = ctx.Builder()->OP_NAME(x, y);                           \
    ctx.AddVar(out_name, out);                                         \
    ctx.AddVarModelToProgram(out_name, out->id);                       \
  }

COMPARE_OPMAPPER_FUNCTION(GreaterThan)
COMPARE_OPMAPPER_FUNCTION(GreaterEqual)
COMPARE_OPMAPPER_FUNCTION(LessThan)
COMPARE_OPMAPPER_FUNCTION(LessEqual)
COMPARE_OPMAPPER_FUNCTION(Equal)
COMPARE_OPMAPPER_FUNCTION(NotEqual)

#undef COMPARE_OPMAPPER_FUNCTION

}  // namespace science_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(science_compare) {
#define COMPARE_OPMAPPER_REGISTER(PD_OP, CINN_OP) \
  CINN_REGISTER_OP_MAPPER(PD_OP,                  \
                          cinn::frontend::science_mappers::CINN_OP##OpMapper)

  COMPARE_OPMAPPER_REGISTER(gt_p, GreaterThan)
  COMPARE_OPMAPPER_REGISTER(ge_p, GreaterEqual)
  COMPARE_OPMAPPER_REGISTER(lt_p, LessThan)
  COMPARE_OPMAPPER_REGISTER(le_p, LessEqual)
  COMPARE_OPMAPPER_REGISTER(eq_p, Equal)
  COMPARE_OPMAPPER_REGISTER(ne_p, NotEqual)

#undef COMPARE_OPMAPPER_REGISTER

  return true;
}
