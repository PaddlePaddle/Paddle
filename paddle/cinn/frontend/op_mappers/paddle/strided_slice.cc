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

#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"
#include "paddle/cinn/frontend/var_type_utils.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace frontend {
namespace paddle_mappers {

void StridedSliceOpMapper(const paddle::cpp::OpDesc& op_desc,
                          const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("Input").size(),
      1UL,
      phi::errors::InvalidArgument("The input of StridedSlice op must be 1."));
  auto x_name = op_desc.Input("Input").front();
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Out").size(),
      1UL,
      phi::errors::InvalidArgument("The output of StridedSlice op must be 1."));
  auto out_name = op_desc.Output("Out").front();

  PADDLE_ENFORCE_EQ(op_desc.HasAttr("starts"),
                    true,
                    phi::errors::InvalidArgument(
                        "StridedSlice op must have starts attribute"));
  auto starts = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "starts"));
  PADDLE_ENFORCE_EQ(
      op_desc.HasAttr("ends"),
      true,
      phi::errors::InvalidArgument("StridedSlice op must have ends attribute"));
  auto ends = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "ends"));
  PADDLE_ENFORCE_EQ(
      op_desc.HasAttr("axes"),
      true,
      phi::errors::InvalidArgument("StridedSlice op must have axes attribute"));
  auto axes = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "axes"));
  PADDLE_ENFORCE_EQ(op_desc.HasAttr("strides"),
                    true,
                    phi::errors::InvalidArgument(
                        "StridedSlice op must have strides attribute"));
  auto strides = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides"));
  PADDLE_ENFORCE_EQ(op_desc.HasAttr("infer_flags"),
                    true,
                    phi::errors::InvalidArgument(
                        "StridedSlice op must have infer_flags attribute"));
  auto infer_flags = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "infer_flags"));
  auto decrease_axis = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "decrease_axis"));

  auto x = ctx.GetVar(x_name);
  auto out = ctx.Builder()->Slice(
      x, axes, starts, ends, infer_flags, strides, decrease_axis);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_strided_slice) {
  CINN_REGISTER_OP_MAPPER(strided_slice,
                          cinn::frontend::paddle_mappers::StridedSliceOpMapper)
  return true;
}
