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

#include <algorithm>

#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/frontend/op_mappers/common_utils.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace frontend {
namespace paddle_mappers {

void ConcatOpMapper(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx) {
  PADDLE_ENFORCE_GE(op_desc.Input("X").size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "The input of concat op must be at least 1."));
  auto x_names = op_desc.Input("X");
  PADDLE_ENFORCE_EQ(
      op_desc.Output("Out").size(),
      1UL,
      phi::errors::InvalidArgument("The output of concat op must be 1."));
  auto out_name = op_desc.Output("Out").front();

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", 0);

  std::vector<Variable> xs;
  for (const auto& name : x_names) {
    xs.emplace_back(ctx.GetVar(name));
  }

  auto err_x = std::find_if(xs.begin(), xs.end(), [&](Variable x) {
    return x->type != xs.front()->type;
  });
  PADDLE_ENFORCE_EQ(err_x == xs.end(),
                    true,
                    phi::errors::InvalidArgument(
                        "All input's dtype of [concat] should be the same."));

  auto out = ctx.Builder()->Concat(xs, axis);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void StackOpMapper(const paddle::cpp::OpDesc& op_desc,
                   const OpMapperContext& ctx) {
  PADDLE_ENFORCE_GE(op_desc.Input("X").size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "The input of stack op must be at least 1."));
  auto x_names = op_desc.Input("X");

  std::string out_name;
  if (op_desc.HasOutput("Out")) {
    PADDLE_ENFORCE_EQ(
        op_desc.Output("Out").size(),
        1UL,
        phi::errors::InvalidArgument("The output of stack op must be 1."));
    out_name = op_desc.Output("Out").front();
  } else if (op_desc.HasOutput("Y")) {
    PADDLE_ENFORCE_EQ(
        op_desc.Output("Y").size(),
        1UL,
        phi::errors::InvalidArgument("The output of stack op must be 1."));
    out_name = op_desc.Output("Y").front();
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "The output argument name of [stack] should be 'Out' or 'Y', "
        "but here cannot found! Please check."));
  }

  cinn::utils::ShapeType input_shape(ctx.GetVar(x_names.front())->shape);
  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", 0);
  axis = axis >= 0 ? axis : axis + input_shape.size() + 1;
  cinn::utils::ShapeType output_shape(input_shape);
  output_shape.insert(output_shape.begin() + axis, 1);

  std::vector<Variable> xs;
  for (const auto& name : x_names) {
    auto x = ctx.GetVar(name);
    PADDLE_ENFORCE_EQ(x->shape == input_shape,
                      true,
                      phi::errors::InvalidArgument(
                          "All input shape of [stack] should be the same."));

    xs.emplace_back(ctx.Builder()->Reshape(x, output_shape));
  }

  auto err_x = std::find_if(xs.begin(), xs.end(), [&](Variable x) {
    return x->type != xs.front()->type;
  });
  PADDLE_ENFORCE_EQ(err_x == xs.end(),
                    true,
                    phi::errors::InvalidArgument(
                        "All input's dtype of [stack] should be the same."));

  auto concat_out = ctx.Builder()->Concat(xs, axis);

  ctx.AddVar(out_name, concat_out);
  ctx.AddVarModelToProgram(out_name, concat_out->id);
}

void SplitOpMapper(const paddle::cpp::OpDesc& op_desc,
                   const OpMapperContext& ctx) {
  PADDLE_ENFORCE_EQ(
      op_desc.Input("X").size(),
      1UL,
      phi::errors::InvalidArgument("The input of split op must be 1."));
  auto x_name = op_desc.Input("X").front();

  PADDLE_ENFORCE_GE(op_desc.Output("Out").size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "The output of split op must be at least 1."));
  auto out_names = op_desc.Output("Out");

  auto x = ctx.GetVar(x_name);
  int rank = x->shape.size();

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", 0);
  CHECK(axis >= -rank && axis < rank)
      << "The [axis] should in [-" << rank << ", " << rank << "), but here is "
      << axis;
  if (axis < 0) {
    axis += rank;
  }

  auto num = utils::GetAttrOrDefault<int>(op_desc, "num", 0);
  auto sections =
      utils::GetAttrOrDefault<std::vector<int>>(op_desc, "sections");

  auto dim = x->shape[axis];
  CHECK(num != 0 || !sections.empty())
      << "The [num_or_sections] in split op should not empty! Please check.";
  if (num != 0) {
    PADDLE_ENFORCE_EQ(
        dim % num == 0,
        true,
        phi::errors::InvalidArgument(
            "The num_or_sections cannot divided by the split axis"));

    sections.clear();
    sections.resize(num, dim / num);
  }
  PADDLE_ENFORCE_EQ(sections.size(),
                    out_names.size(),
                    phi::errors::InvalidArgument(
                        "The output number of split op should be same"));

  int neg_idx = -1, sum = 0;
  for (int i = 0; i < sections.size(); ++i) {
    if (sections[i] < 0) {
      PADDLE_ENFORCE_LT(
          neg_idx,
          0,
          phi::errors::InvalidArgument(
              "The [num_or_sections] should only has one -1! But here "
              "found more than one."));
      neg_idx = i;
    } else {
      sum += sections[i];
    }
  }
  if (neg_idx > 0) {
    PADDLE_ENFORCE_LT(sum,
                      dim,
                      phi::errors::InvalidArgument(
                          "The sum of [num_or_sections] should less than to "
                          "the dimension of split [axis] when -1 "
                          "found in [num_or_sections]! But here "
                          "found more than one."));
    sections[neg_idx] = dim - sum;
  } else {
    PADDLE_ENFORCE_EQ(sum,
                      dim,
                      phi::errors::InvalidArgument(
                          "The sum of [num_or_sections] should equal to the "
                          "dimension of split [axis]! But here "
                          "found more than one."));
  }

  auto outs = ctx.Builder()->Split(x, sections, axis);

  for (int i = 0; i < out_names.size(); ++i) {
    ctx.AddVar(out_names[i], outs[i]);
    ctx.AddVarModelToProgram(out_names[i], outs[i]->id);
  }
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_concat) {
  CINN_REGISTER_OP_MAPPER(concat,
                          cinn::frontend::paddle_mappers::ConcatOpMapper)
  CINN_REGISTER_OP_MAPPER(stack, cinn::frontend::paddle_mappers::StackOpMapper)
  CINN_REGISTER_OP_MAPPER(split, cinn::frontend::paddle_mappers::SplitOpMapper)
  return true;
}
