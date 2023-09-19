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

#include "paddle/cinn/frontend/decomposer_registry.h"
#include "paddle/cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {
namespace decomposer {

void relu(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 1UL)
      << " 1 input tensor for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 1UL)
      << "1 output tensor for " << instr->op_type;
  auto x = instr->inputs[0];
  auto output = instr->outputs[0];
  auto* builder = context.builder();

  auto bcast_zero = builder->FillConstant(
      x->shape, 0.0f, common::UniqName("zero"), common::Type2Str(x->type));
  auto out = builder->Max(x, bcast_zero);

  // map the the output of decomposed operator to the original.
  context.MapOutToOrigin(out, output);
}

void relu_grad(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 2UL)
      << " 2 input tensors for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 1UL)
      << "1 output tensor for " << instr->op_type;
  auto dout = instr->inputs[0];
  auto out = instr->inputs[1];
  auto dx = instr->outputs[0];
  auto* builder = context.builder();

  auto bcast_zero = builder->FillConstant(
      out->shape, 0.0f, common::UniqName("zero"), common::Type2Str(out->type));
  auto condition = builder->GreaterThan(out, bcast_zero);
  auto res = builder->Select(condition, dout, bcast_zero);

  // map the the output of decomposed operator to the original.
  context.MapOutToOrigin(res, dx);
}

void gelu(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 1UL)
      << " 1 input tensor for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 1UL)
      << "1 output tensor for " << instr->op_type;
  auto x = instr->inputs[0];
  auto output = instr->outputs[0];
  auto* builder = context.builder();

  // x * (0.5 + 0.5 * erf(sqrtf(0.5) * x))
  auto p_5 = builder->FillConstant(
      x->shape, 0.5f, common::UniqName("p_5"), common::Type2Str(x->type));
  auto p_7 = builder->FillConstant(x->shape,
                                   std::sqrt(0.5),
                                   common::UniqName("p_7"),
                                   common::Type2Str(x->type));
  auto erf = builder->Erf(builder->Multiply(x, p_7));
  auto cdf = builder->Add(p_5, builder->Multiply(p_5, erf));
  auto out = builder->Multiply(x, cdf);

  // map the the output of decomposed operator to the original.
  context.MapOutToOrigin(out, output);
}

void softmax(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 1UL)
      << " 1 input tensor for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 1UL)
      << "1 output tensor for " << instr->op_type;
  auto x = instr->inputs[0];
  auto output = instr->outputs[0];
  auto* builder = context.builder();

  std::vector<int> b_axes;
  auto axes = instr.GetAttrs<std::vector<int>>("axes");
  CHECK(axes.size());
  for (auto& axis : axes) {
    if (axis < 0) {
      axis += x->shape.size();
    }
  }
  for (int idx = 0; idx < x->shape.size(); ++idx) {
    if (std::find(axes.begin(), axes.end(), idx) == axes.end()) {
      b_axes.push_back(idx);
    }
  }

  // When the rank of x is 1, broadcast axes will be empty, so we need to insert
  // last dim as broadcast axis.
  if (b_axes.empty()) {
    b_axes.emplace_back(-1);
  }

  auto mode = instr.GetAttrs<std::string>("mode");
  if (mode == "fast") {
    // x_sum = sum(exp(x))
    auto x_sum = builder->BroadcastTo(
        builder->ReduceSum(builder->Exp(x), axes), x->shape, b_axes);
    // x_exp / x_sum
    auto out = builder->Divide(builder->Exp(x), x_sum);

    // map the the output of decomposed operator to the original.
    context.MapOutToOrigin(out, output);
  } else {
    // x = max(x)
    auto x_max =
        builder->BroadcastTo(builder->ReduceMax(x, axes), x->shape, b_axes);
    // x_exp = exp(x - x_max)
    auto x_exp = builder->Exp(builder->Subtract(x, x_max));
    // x_sum = sum(x_exp)
    auto x_sum =
        builder->BroadcastTo(builder->ReduceSum(x_exp, axes), x->shape, b_axes);
    // x_exp / x_sum
    auto out =
        builder->Divide(builder->Exp(builder->Subtract(x, x_max)), x_sum);

    // map the the output of decomposed operator to the original.
    context.MapOutToOrigin(out, output);
  }
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(relu_decomposers) {
  CINN_DECOMPOSER_REGISTER(relu, cinn::frontend::decomposer::relu);

  return true;
}

CINN_REGISTER_HELPER(relu_grad_decomposers) {
  CINN_DECOMPOSER_REGISTER(relu_grad, cinn::frontend::decomposer::relu_grad);

  return true;
}

CINN_REGISTER_HELPER(gelu_decomposers) {
  CINN_DECOMPOSER_REGISTER(gelu, cinn::frontend::decomposer::gelu);

  return true;
}

CINN_REGISTER_HELPER(softmax_decomposers) {
  CINN_DECOMPOSER_REGISTER(softmax, cinn::frontend::decomposer::softmax);

  return true;
}
