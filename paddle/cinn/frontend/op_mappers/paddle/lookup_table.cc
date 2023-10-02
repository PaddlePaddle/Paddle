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

void LookupTableOpMapper(const paddle::cpp::OpDesc& op_desc,
                         const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("W").size(), 1UL);
  auto w_name = op_desc.Input("W").front();
  CHECK_EQ(op_desc.Input("Ids").size(), 1UL);
  auto ids_name = op_desc.Input("Ids").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  auto w = ctx.GetVar(w_name);
  auto ids = ctx.GetVar(ids_name);
  CHECK(op_desc.HasAttr("padding_idx"));
  auto padding_idx =
      utils::GetAttrOrDefault<int64_t>(op_desc, "padding_idx", -1);
  auto out = ctx.Builder()->LookupTable(w, ids, padding_idx);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void LookupTableV2OpMapper(const paddle::cpp::OpDesc& op_desc,
                           const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("W").size(), 1UL);
  auto w_name = op_desc.Input("W").front();
  CHECK_EQ(op_desc.Input("Ids").size(), 1UL);
  auto ids_name = op_desc.Input("Ids").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  auto w = ctx.GetVar(w_name);
  auto ids = ctx.GetVar(ids_name);
  ids = ctx.Builder()->ExpandDims(ids, {-1});
  CHECK(op_desc.HasAttr("padding_idx"));
  auto padding_idx =
      utils::GetAttrOrDefault<int64_t>(op_desc, "padding_idx", -1);
  auto out = ctx.Builder()->LookupTable(w, ids, padding_idx);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_lookup_table) {
  CINN_REGISTER_OP_MAPPER(lookup_table,
                          cinn::frontend::paddle_mappers::LookupTableOpMapper)
  CINN_REGISTER_OP_MAPPER(lookup_table_v2,
                          cinn::frontend::paddle_mappers::LookupTableV2OpMapper)
  return true;
}
