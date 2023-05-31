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

void sum(const Instruction& instr, const DecomposerContext& context) {
  CHECK_GT(instr->inputs.size(), 0UL)
      << "At least 1 input tensor for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 1UL)
      << "1 output tensor for " << instr->op_type;
  auto inputs = instr->inputs;
  auto output = instr->outputs[0];
  auto* builder = context.builder();

  auto sum = builder->Identity(inputs[0]);
  for (size_t i = 1; i < inputs.size(); ++i) {
    sum = builder->Add(sum, inputs[i]);
  }

  // map the the output of decomposed operator to the original.
  context.MapOutToOrigin(sum, output);
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(sum_decomposers) {
  CINN_DECOMPOSER_REGISTER(sum, cinn::frontend::decomposer::sum);

  return true;
}
