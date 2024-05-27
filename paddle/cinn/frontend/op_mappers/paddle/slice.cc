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
#include "paddle/common/enforce.h"
namespace cinn {
namespace frontend {
namespace paddle_mappers {

void SliceOpMapper(const paddle::cpp::OpDesc& op_desc,
                   const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("Input").size(),
      1UL,
      phi::errors::InvalidArgument("The input of Slice op must be 1."));
  auto x_name = op_desc.Input("Input").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Out").size(),
      1UL,
      phi::errors::InvalidArgument("The output of Slice op must be 1."));
  auto out_name = op_desc.Output("Out").front();

  PADDLE_ENFORCE_EQ(
      op_desc.HasAttr("starts"),
      true,
      phi::errors::InvalidArgument("Slice op must have starts attribute"));
  auto starts = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "starts");
  PADDLE_ENFORCE_EQ(
      op_desc.HasAttr("ends"),
      true,
      phi::errors::InvalidArgument("Slice op must have ends attribute"));
  auto ends = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "ends");
  PADDLE_ENFORCE_EQ(
      op_desc.HasAttr("axes"),
      true,
      phi::errors::InvalidArgument("Slice op must have axes attribute"));
  auto axes = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "axes");

  auto infer_flags =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "infer_flags");
  auto strides = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides");
  auto decrease_axis =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "decrease_axis");
  auto x = ctx.GetVar(x_name);
  auto out = ctx.Builder()->Slice(
      x, axes, starts, ends, infer_flags, strides, decrease_axis);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_slice) {
  CINN_REGISTER_OP_MAPPER(slice, cinn::frontend::paddle_mappers::SliceOpMapper)
  return true;
}
