// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/type.h"
#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace frontend {
namespace paddle_mappers {

void ScatterOpMapper(const paddle::cpp::OpDesc& op_desc,
                     const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of Scatter op must be 1."));
  auto x_name = op_desc.Input("X").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Input("Ids").size(),
      1UL,
      phi::errors::InvalidArgument("The input of Scatter op must be 1."));
  auto ids_name = op_desc.Input("Ids").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Input("Updates").size(),
      1UL,
      phi::errors::InvalidArgument("The input of Scatter op must be 1."));
  auto updates_name = op_desc.Input("Updates").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Out").size(),
      1UL,
      phi::errors::InvalidArgument("The output of Scatter op must be 1."));
  auto out_name = op_desc.Output("Out").front();

  bool overwrite = utils::GetAttrOrDefault<bool>(op_desc, "overwrite", true);
  VLOG(4) << "out_name = scatter(X=" << x_name << ", Ids=" << ids_name
          << ", Updates=" << updates_name << ", overwrite=" << overwrite << ")";

  const auto& input = ctx.GetVar(x_name);
  auto indices = ctx.GetVar(ids_name);
  const auto& updates = ctx.GetVar(updates_name);
  PADDLE_ENFORCE_EQ(input->type == updates->type,
                    true,
                    phi::errors::InvalidArgument(
                        "The type of input and updates should be the same."));
  CHECK(indices->type == cinn::common::Int(32) ||
        indices->type == cinn::common::Int(64))
      << "checks whether the data type of the indices is either int32 or int64";
  if (indices->type == cinn::common::Int(64)) {
    indices = ctx.Builder()->Cast(
        indices, cinn::common::Type2Str(cinn::common::Int(32)));
  }
  PADDLE_ENFORCE_LE(indices->shape.size(),
                    2UL,
                    phi::errors::InvalidArgument(
                        "The rank of indices should be less than 2."));
  if (indices->shape.size() == 0) {
    indices = ctx.Builder()->Reshape(indices, {1});
  }
  if (indices->shape.size() == 2) {
    indices = ctx.Builder()->Reshape(indices,
                                     {indices->shape[0] * indices->shape[1]});
  }

  Variable out;
  if (overwrite) {
    out = ctx.Builder()->ScatterAssign(input, updates, indices);
  } else {
    const auto& zeros =
        ctx.Builder()->FillConstant(updates->shape,
                                    0,
                                    cinn::common::UniqName("scatter_zeros"),
                                    cinn::common::Type2Str(updates->type));
    out = ctx.Builder()->ScatterAssign(input, zeros, indices);
    out = ctx.Builder()->ScatterAdd(out, updates, indices);
  }

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_scatter) {
  CINN_REGISTER_OP_MAPPER(scatter,
                          cinn::frontend::paddle_mappers::ScatterOpMapper)
  return true;
}
