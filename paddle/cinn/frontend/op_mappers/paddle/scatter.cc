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

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void ScatterOpMapper(const paddle::cpp::OpDesc& op_desc,
                     const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Ids").size(), 1UL);
  auto ids_name = op_desc.Input("Ids").front();
  CHECK_EQ(op_desc.Input("Updates").size(), 1UL);
  auto updates_name = op_desc.Input("Updates").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  bool overwrite = utils::GetAttrOrDefault<bool>(op_desc, "overwrite", true);
  VLOG(4) << "out_name = scatter(X=" << x_name << ", Ids=" << ids_name
          << ", Updates=" << updates_name << ", overwrite=" << overwrite << ")";

  const auto& input = ctx.GetVar(x_name);
  auto indices = ctx.GetVar(ids_name);
  const auto& updates = ctx.GetVar(updates_name);
  CHECK(input->type == updates->type)
      << "checks whether the type of the input and the updates are the same.";
  CHECK(indices->type == common::Int(32) || indices->type == common::Int(64))
      << "checks whether the data type of the indices is either int32 or int64";
  if (indices->type == common::Int(64)) {
    indices = ctx.Builder()->Cast(indices, common::Type2Str(common::Int(32)));
  }
  CHECK_LE(indices->shape.size(), 2) << "Ids should be 0, 1 or 2 in scatter_op";
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
                                    common::UniqName("scatter_zeros"),
                                    common::Type2Str(updates->type));
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
