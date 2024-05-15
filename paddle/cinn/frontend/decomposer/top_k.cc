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

#include "paddle/cinn/frontend/decomposer_registry.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace frontend {
namespace decomposer {

void top_k(const Instruction& instr, const DecomposerContext& context) {
  PADDLE_ENFORCE_EQ(
      instr->inputs.size(),
      1UL,
      phi::errors::InvalidArgument("The input tensor size should be 1."));
  PADDLE_ENFORCE_EQ(
      instr->outputs.size(),
      2UL,
      phi::errors::InvalidArgument("The output tensor size should be 2."));
  auto x = instr->inputs[0];
  auto output = instr->outputs[0];
  auto indices = instr->outputs[1];

  auto* builder = context.builder();
  int k = instr.GetAttrs<int>("k");
  PADDLE_ENFORCE_GT(
      k,
      0,
      phi::errors::InvalidArgument("The attribute k must be greater than 0."));
  int axis = instr.GetAttrs<int>("axis");
  if (axis < 0) {
    axis += x->shape.size();
  }

  auto sort_tmp = builder->Sort(x, axis, false);
  auto sort_out = builder->Slice(sort_tmp, {axis}, {0}, {k});
  auto argsort_tmp = builder->ArgSort(x, axis, false).at(0);
  auto argsort_out =
      builder->Cast(builder->Slice(argsort_tmp, {axis}, {0}, {k}), "int64");

  // map the the output of decomposed operator to the original.
  context.MapOutToOrigin(sort_out, output);
  context.MapOutToOrigin(argsort_out, indices);
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(top_k_decomposer) {
  CINN_DECOMPOSER_REGISTER(top_k, cinn::frontend::decomposer::top_k);
  return true;
}
