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

void RollOpMapper(const paddle::cpp::OpDesc& op_desc,
                  const OpMapperContext& ctx) {
  // input
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  // output
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  // attr shifts and axis
  CHECK(op_desc.HasAttr("shifts"));
  CHECK(op_desc.HasAttr("axis"));
  std::vector<int> shifts = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "shifts", {1}));
  std::vector<int> axis = utils::ToShapeType(
      utils::GetAttrOrDefault<std::vector<int64_t>>(op_desc, "axis", {}));

  auto x = ctx.GetVar(x_name);
  auto vec_x_dims = std::vector<int>(x->shape);
  std::vector<int> output_shape = vec_x_dims;

  // check axis and shifts and when axis is None, we should flatten x.
  bool axis_None = false;
  if (axis.size() == 0) {
    CHECK_EQ(shifts.size(), 1)
        << "shifts.size() should be equal to 1 when axis is None";
    axis.push_back(0);
    axis_None = true;
    int reshape_num = 1;
    for (int i = 0; i < vec_x_dims.size(); ++i) {
      reshape_num *= vec_x_dims[i];
    }
    vec_x_dims = std::vector<int>{reshape_num};
    x = ctx.Builder()->Reshape(x, vec_x_dims);
  } else {
    CHECK_EQ(shifts.size(), axis.size())
        << "shifts.size() should be equal to axis.size()";
  }

  // preprocessing the shifts and axis
  int shifts_size = shifts.size();
  std::unordered_map<int, int> axis_to_shifts;
  for (int i = 0; i < shifts_size; ++i) {
    int vec_x_dims_size = vec_x_dims.size();
    CHECK_GE(axis[i], -vec_x_dims_size)
        << "axis value should be >= " << -vec_x_dims_size;
    CHECK_LT(axis[i], vec_x_dims_size)
        << "axis value should be < " << vec_x_dims_size;
    if (axis[i] < 0) {
      axis[i] += vec_x_dims_size;
    }
    // optimize for the same axis
    if (axis_to_shifts.count(axis[i]) > 0) {
      axis_to_shifts[axis[i]] += shifts[i];
    } else {
      axis_to_shifts[axis[i]] = shifts[i];
    }
    int size = vec_x_dims[axis[i]];
    if (size > 0) {
      axis_to_shifts[axis[i]] = (axis_to_shifts[axis[i]] % size + size) % size;
    }
  }

  auto output = ctx.Builder()->Identity(x);
  // use Split + Concat for each shift
  for (const auto& pair : axis_to_shifts) {
    if (pair.second > 0) {
      int length = vec_x_dims[pair.first];
      auto front_slice = ctx.Builder()->Slice(
          output, {pair.first}, {0}, {length - pair.second});
      auto behind_slice = ctx.Builder()->Slice(
          output, {pair.first}, {length - pair.second}, {length});
      auto split_output = std::vector<Variable>{behind_slice, front_slice};
      output = ctx.Builder()->Concat(split_output, pair.first);
    }
  }

  if (axis_None) {
    output = ctx.Builder()->Reshape(output, output_shape);
  }

  ctx.AddVar(out_name, output);
  ctx.AddVarModelToProgram(out_name, output->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_roll) {
  CINN_REGISTER_OP_MAPPER(roll, cinn::frontend::paddle_mappers::RollOpMapper)
  return true;
}
