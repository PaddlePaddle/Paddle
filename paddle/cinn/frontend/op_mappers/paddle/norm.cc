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

struct NormHelper {
  NormHelper(NetBuilder* net_builder, int32_t axis) {
    builder = net_builder;
    reduce_dim = {axis};
    num_instructions = builder->size();
  }

  ~NormHelper() {
    VLOG(4) << "norm is decomposed to " << builder->size() - num_instructions
            << " instructions.";
  }

  // square_sum = reduce_sum(x * x)
  Variable SquareSum(Variable x) {
    auto x_square = builder->Multiply(x, builder->Identity(x));
    auto x_square_sum = Reduce(x_square);

    return x_square_sum;
  }

  // std_square_sum = sqrt(square_sum + epsilon)
  Variable StdSquareSum(Variable square_sum, float epsilon) {
    auto epsilon_1d = builder->FillConstant(square_sum->shape,
                                            epsilon,
                                            common::UniqName("norm_epsilon"),
                                            common::Type2Str(square_sum->type));
    auto std_square_sum = builder->Sqrt(builder->Add(square_sum, epsilon_1d));
    return std_square_sum;
  }

  Variable Reduce(Variable x) {
    return builder->ReduceSum(x, reduce_dim, true);
  }

  NetBuilder* builder{nullptr};
  std::vector<int> reduce_dim;
  int num_instructions{0};
};

void NormOpMapper(const paddle::cpp::OpDesc& op_desc,
                  const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  std::string norm_name;
  if (op_desc.HasOutput("Norm") && !op_desc.Output("Norm").empty()) {
    CHECK_EQ(op_desc.Output("Norm").size(), 1UL);
    norm_name = op_desc.Output("Norm").front();
  }

  CHECK(op_desc.HasAttr("axis"));
  auto axis = utils::GetAttrOrDefault<int>(op_desc, "axis", -1);
  auto epsilon = utils::GetAttrOrDefault<float>(op_desc, "epsilon", 1.0e-10f);
  auto is_test =
      utils::GetAttrOrDefault<bool>(op_desc, "is_test", norm_name.empty());

  auto x = ctx.GetVar(x_name);

  VLOG(4) << "Out=" << out_name << ", Norm=" << norm_name
          << " = norm(X:" << x_name << "=" << x << ", axis=" << axis
          << ", epsilon=" << epsilon << ", is_test=" << std::ios::boolalpha
          << is_test;

  if (axis < 0) {
    axis += x->shape.size();
  }
  CHECK_GE(axis, 0);
  CHECK_LT(axis, x->shape.size());

  NormHelper helper(ctx.Builder(), axis);

  auto in_type = x->type;
  if (in_type.is_float16() || in_type.is_bfloat16()) {
    x = ctx.Builder()->Cast(x, "float32");
  }
  auto square_sum = helper.SquareSum(x);
  auto std_square_sum = helper.StdSquareSum(square_sum, epsilon);
  auto normalized = ctx.Builder()->Divide(x, std_square_sum);
  auto y = ctx.Builder()->Cast(normalized, common::Type2Str(in_type));

  ctx.AddVar(out_name, y);
  ctx.AddVarModelToProgram(out_name, y->id);

  if (!norm_name.empty()) {
    auto norm_grad =
        ctx.Builder()->Cast(std_square_sum, common::Type2Str(in_type));
    ctx.AddVar(norm_name, norm_grad);
    ctx.AddVarModelToProgram(norm_name, norm_grad->id);
  }
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_norm) {
  CINN_REGISTER_OP_MAPPER(norm, cinn::frontend::paddle_mappers::NormOpMapper)
  return true;
}
