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

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void CumsumOpMapper(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();

  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto input = ctx.GetVar(x_name);
  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", -1);
  auto flatten = utils::GetAttrOrDefault<bool>(op_desc, "flatten", false);
  auto exclusive = utils::GetAttrOrDefault<bool>(op_desc, "exclusive", false);
  auto reverse = utils::GetAttrOrDefault<bool>(op_desc, "reverse", false);

  auto x = input;
  int ndim = x->shape.size();
  // If flatten = true, flatten x and do cumsum on axis 0.
  if (flatten) {
    x = ctx.Builder()->Reshape(x, {-1});
    axis = 0;
    ndim = x->shape.size();
  }
  CHECK(-ndim <= axis && axis < ndim)
      << "Axis expected to be in range of [" << -ndim << "," << ndim
      << "]. But got " << axis << ".";
  if (axis < 0) {
    axis = ndim + axis;
  }
  x = ctx.Builder()->ExpandDims(x, {axis + 1});
  auto rg = ctx.Builder()->Arange(
      0.0f, static_cast<float>(x->shape[axis]), 1.0f, "int32");
  cinn::frontend::Variable mask;
  if (reverse) {
    mask = ctx.Builder()->GreaterEqual(ctx.Builder()->ExpandDims(rg, {1}), rg);
  } else {
    mask = ctx.Builder()->LessEqual(ctx.Builder()->ExpandDims(rg, {1}), rg);
  }
  for (int i = 0; i < ndim - axis - 1; i++) {
    mask = ctx.Builder()->ExpandDims(mask, {-1});
  }
  // Infer broadcast shape for x and mask
  int x_ndim = x->shape.size();
  int mask_ndim = mask->shape.size();
  std::vector<int> broadcast_shape(std::max(x_ndim, mask_ndim), 0);
  int broadcast_shape_size = broadcast_shape.size();
  for (int i = broadcast_shape.size() - 1; i >= 0; --i) {
    if (i - (broadcast_shape_size - x_ndim) >= 0) {
      broadcast_shape[i] = std::max(
          broadcast_shape[i], x->shape[i - (broadcast_shape_size - x_ndim)]);
    }
    if (i - (broadcast_shape_size - mask_ndim) >= 0) {
      broadcast_shape[i] =
          std::max(broadcast_shape[i],
                   mask->shape[i - (broadcast_shape_size - mask_ndim)]);
    }
  }
  // Do broadcast shape on mask, x and false_value
  mask = ctx.Builder()->BroadcastTo(mask, broadcast_shape);
  x = ctx.Builder()->BroadcastTo(x, broadcast_shape);
  auto false_value = ctx.Builder()->FillConstant(
      x->shape, 0, UniqName("false_value"), common::Type2Str(x->type));
  // Select elements with mask
  auto selected_x = ctx.Builder()->Select(mask, x, false_value);
  // Do reduce sum
  auto output = ctx.Builder()->ReduceSum(selected_x, {axis});
  // Exclusive
  if (exclusive) {
    output = ctx.Builder()->Subtract(output, input);
  }
  ctx.AddVar(out_name, output);
  ctx.AddVarModelToProgram(out_name, output->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_cumsum) {
  CINN_REGISTER_OP_MAPPER(cumsum,
                          cinn::frontend::paddle_mappers::CumsumOpMapper)
  return true;
}
