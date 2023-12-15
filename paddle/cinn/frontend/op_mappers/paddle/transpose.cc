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

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void TransposeOpMapper(const paddle::cpp::OpDesc& op_desc,
                       const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  auto x = ctx.GetVar(x_name);

  auto axis = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "axis");

  VLOG(4) << "x shape: " << cinn::utils::Join(x->shape, ",");
  VLOG(4) << "transpose perm : " << cinn::utils::Join(axis, ",");

  auto out = ctx.Builder()->Transpose(x, axis);

  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void Transpose2OpMapper(const paddle::cpp::OpDesc& op_desc,
                        const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  auto x = ctx.GetVar(x_name);

  auto axis = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "axis");

  VLOG(4) << "x shape: " << cinn::utils::Join(x->shape, ",");
  VLOG(4) << "transpose2 perm : " << cinn::utils::Join(axis, ",");

  auto out = ctx.Builder()->Transpose(x, axis);

  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);

  if (op_desc.HasOutput("XShape")) {
    // transpose2 adds an intermediate output(XShape) based on
    // transpose, the XShape is used to carry the shape and lod of X which
    // will be used in transpose_grad, in this way, the framework can reuse
    // the memory of X immediately the transpose2_op is finished.
    // Considering compatibility issues, we could not fix transpose2_op
    CHECK_EQ(op_desc.Output("XShape").size(), 1UL);
    auto xshape_name = op_desc.Output("XShape").front();

    auto xshape = ctx.Builder()->Identity(x);

    ctx.AddVar(xshape_name, xshape);
    ctx.AddVarModelToProgram(xshape_name, xshape->id);
  }
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_transpose) {
  CINN_REGISTER_OP_MAPPER(transpose,
                          cinn::frontend::paddle_mappers::TransposeOpMapper)
  CINN_REGISTER_OP_MAPPER(transpose2,
                          cinn::frontend::paddle_mappers::Transpose2OpMapper)
  return true;
}
