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

void MatMulOpMapper(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto trans_x = utils::GetAttrOrDefault<bool>(op_desc, "trans_x", false);
  trans_x = utils::GetAttrOrDefault<bool>(op_desc, "transpose_X", trans_x);

  auto trans_y = utils::GetAttrOrDefault<bool>(op_desc, "trans_y", false);
  trans_y = utils::GetAttrOrDefault<bool>(op_desc, "transpose_Y", trans_y);

  auto alpha = utils::GetAttrOrDefault<float>(op_desc, "alpha", 1.0f);

  VLOG(4) << out_name << "=matmul{" << x_name << ", " << y_name
          << ", trans_x=" << trans_x << ", trans_y=" << trans_y
          << ", alpha=" << alpha << "}";

  auto x = ctx.GetVar(x_name);
  auto y = ctx.GetVar(y_name);
  auto out = ctx.Builder()->Matmul(x, y, trans_x, trans_y, alpha);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_matmul) {
  CINN_REGISTER_OP_MAPPER(matmul,
                          cinn::frontend::paddle_mappers::MatMulOpMapper)
  CINN_REGISTER_OP_MAPPER(matmul_v2,
                          cinn::frontend::paddle_mappers::MatMulOpMapper)
  return true;
}
