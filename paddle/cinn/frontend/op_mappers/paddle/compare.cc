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

static const std::string& GetCompareDebugString(const std::string& compare_op) {
  static std::unordered_map<std::string, std::string> compare_debug_map = {
      {"GreaterThan", " > "},
      {"GreaterEqual", " >= "},
      {"LessThan", " < "},
      {"LessEqual", " <= "},
      {"Equal", " == "},
      {"NotEqual", " != "},
  };
  CHECK_GT(compare_debug_map.count(compare_op), 0)
      << "Unsupported compare op " << compare_op;
  return compare_debug_map[compare_op];
}

#define COMPARE_OPMAPPER_FUNCTION(OP_NAME)                                    \
  void OP_NAME##OpMapper(const paddle::cpp::OpDesc& op_desc,                  \
                         const OpMapperContext& ctx) {                        \
    CHECK_EQ(op_desc.Input("X").size(), 1UL);                                 \
    auto x_name = op_desc.Input("X").front();                                 \
    CHECK_EQ(op_desc.Input("Y").size(), 1UL);                                 \
    auto y_name = op_desc.Input("Y").front();                                 \
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);                              \
    auto out_name = op_desc.Output("Out").front();                            \
    auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", -1);            \
    VLOG(4) << out_name << " = " << x_name << GetCompareDebugString(#OP_NAME) \
            << y_name << " at " << axis;                                      \
    auto x = ctx.GetVar(x_name);                                              \
    auto y = ctx.GetVar(y_name);                                              \
    auto out = ctx.Builder()->OP_NAME(x, y, axis);                            \
    ctx.AddVar(out_name, out);                                                \
    ctx.AddVarModelToProgram(out_name, out->id);                              \
  }

COMPARE_OPMAPPER_FUNCTION(GreaterThan)
COMPARE_OPMAPPER_FUNCTION(GreaterEqual)
COMPARE_OPMAPPER_FUNCTION(LessThan)
COMPARE_OPMAPPER_FUNCTION(LessEqual)
COMPARE_OPMAPPER_FUNCTION(Equal)
COMPARE_OPMAPPER_FUNCTION(NotEqual)

#undef COMPARE_OPMAPPER_FUNCTION

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_compare) {
#define COMPARE_OPMAPPER_REGISTER(PD_OP, CINN_OP) \
  CINN_REGISTER_OP_MAPPER(PD_OP,                  \
                          cinn::frontend::paddle_mappers::CINN_OP##OpMapper)

  COMPARE_OPMAPPER_REGISTER(greater_than, GreaterThan)
  COMPARE_OPMAPPER_REGISTER(greater_equal, GreaterEqual)
  COMPARE_OPMAPPER_REGISTER(less_than, LessThan)
  COMPARE_OPMAPPER_REGISTER(less_equal, LessEqual)
  COMPARE_OPMAPPER_REGISTER(equal, Equal)
  COMPARE_OPMAPPER_REGISTER(not_equal, NotEqual)

#undef COMPARE_OPMAPPER_REGISTER

  return true;
}
