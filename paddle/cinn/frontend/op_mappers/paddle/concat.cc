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

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void ConcatOpMapper(const paddle::cpp::OpDesc& op_desc,
                    const OpMapperContext& ctx) {
  CHECK_GE(op_desc.Input("X").size(), 1UL);
  auto x_names = op_desc.Input("X");
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", 0);

  std::vector<Variable> xs;
  for (const auto& name : x_names) {
    xs.emplace_back(ctx.GetVar(name));
  }

  auto err_x = std::find_if(xs.begin(), xs.end(), [&](Variable x) {
    return x->type != xs.front()->type;
  });
  CHECK(err_x == xs.end())
      << "All input's dtype of [concat] should be the same, be the input "
      << (*err_x)->id << "'s dtype [" << (*err_x)->type
      << "] not equal to the first input " << xs.front()->id << "'s dtype ["
      << xs.front()->type << "]";

  auto out = ctx.Builder()->Concat(xs, axis);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void StackOpMapper(const paddle::cpp::OpDesc& op_desc,
                   const OpMapperContext& ctx) {
  CHECK_GE(op_desc.Input("X").size(), 1UL);
  auto x_names = op_desc.Input("X");

  std::string out_name;
  if (op_desc.HasOutput("Out")) {
    CHECK_EQ(op_desc.Output("Out").size(), 1UL);
    out_name = op_desc.Output("Out").front();
  } else if (op_desc.HasOutput("Y")) {
    CHECK_EQ(op_desc.Output("Y").size(), 1UL);
    out_name = op_desc.Output("Y").front();
  } else {
    LOG(FATAL) << "The output argument name of [stack] should be 'Out' or 'Y', "
                  "but here cannot found! Please check.";
  }

  cinn::utils::ShapeType input_shape(ctx.GetVar(x_names.front())->shape);
  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", 0);
  axis = axis >= 0 ? axis : axis + input_shape.size() + 1;
  cinn::utils::ShapeType output_shape(input_shape);
  output_shape.insert(output_shape.begin() + axis, 1);

  std::vector<Variable> xs;
  for (const auto& name : x_names) {
    auto x = ctx.GetVar(name);
    CHECK(x->shape == input_shape)
        << "All input shape of [stack] should be the same, be the input "
        << x->id << "'s shape [" << cinn::utils::Join(x->shape, ", ")
        << "] not equal to "
        << "the first input " << ctx.GetVar(x_names.front())->id << "'s shape ["
        << cinn::utils::Join(input_shape, ", ") << "]";

    xs.emplace_back(ctx.Builder()->Reshape(x, output_shape));
  }

  auto err_x = std::find_if(xs.begin(), xs.end(), [&](Variable x) {
    return x->type != xs.front()->type;
  });
  CHECK(err_x == xs.end())
      << "All input's dtype of [concat] should be the same, be the input "
      << (*err_x)->id << "'s dtype [" << (*err_x)->type
      << "] not equal to the first input " << xs.front()->id << "'s dtype ["
      << xs.front()->type << "]";

  auto concat_out = ctx.Builder()->Concat(xs, axis);

  ctx.AddVar(out_name, concat_out);
  ctx.AddVarModelToProgram(out_name, concat_out->id);
}

void SplitOpMapper(const paddle::cpp::OpDesc& op_desc,
                   const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();

  CHECK_GE(op_desc.Output("Out").size(), 1UL);
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
    CHECK(dim % num == 0) << "The num_or_sections:" << num
                          << " cannot divided by the split axis:" << axis
                          << " 's dimension:" << dim;

    sections.clear();
    sections.resize(num, dim / num);
  }
  CHECK_EQ(sections.size(), out_names.size())
      << "The output number of split op should be " << sections.size()
      << ", but actual " << out_names.size();

  int neg_idx = -1, sum = 0;
  for (int i = 0; i < sections.size(); ++i) {
    if (sections[i] < 0) {
      CHECK_LT(neg_idx, 0)
          << "The [num_or_sections] should only has one -1! But here "
          << cinn::utils::Join(sections, ", ");
      neg_idx = i;
    } else {
      sum += sections[i];
    }
  }
  if (neg_idx > 0) {
    CHECK_LT(sum, dim) << "The sum of [num_or_sections] should less than to "
                          "the dimension of split [axis] when -1 "
                          "found in [num_or_sections]! But here "
                       << cinn::utils::Join(sections, ", ");
    sections[neg_idx] = dim - sum;
  } else {
    CHECK_EQ(sum, dim) << "The sum of [num_or_sections] should equal to the "
                          "dimension of split [axis]! But here "
                       << cinn::utils::Join(sections, ", ");
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
