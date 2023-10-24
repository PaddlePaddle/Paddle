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

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "glog/logging.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/program_pass.h"

namespace cinn {
namespace frontend {
namespace pass {

namespace {
using CastImplFunc =
    std::function<void(NetBuilder* builder, const Instruction&)>;

bool IsInputHasFP16OrBF16(const std::vector<Variable>& inputs) {
  return std::find_if(inputs.begin(), inputs.end(), [](const Variable& var) {
           return var->type.is_float16() || var->type.is_bfloat16();
         }) != inputs.end();
}

Instruction CreateNewCastInstruction(const Variable& input,
                                     const Variable& output) {
  Instruction new_cast_instr("cast", {input});
  new_cast_instr->outputs = {output};
  new_cast_instr->attrs = {{"dtype", common::Type2Str(output->type)}};
  new_cast_instr->attrs_ordered = {{"dtype", common::Type2Str(output->type)}};
  return new_cast_instr;
}

Instruction CreateNewIdentityInstruction(const Variable& input,
                                         const Variable& output) {
  Instruction new_identity_instr("identity", {input});
  new_identity_instr->outputs = {output};
  return new_identity_instr;
}

void CommonCastImpl(NetBuilder* builder, const Instruction& instr) {
  if (!IsInputHasFP16OrBF16(instr->inputs)) {
    // DO NOT NEED CAST
    builder->AppendInstruction(instr);
    return;
  }

  // Cast all fp16/bf16 inputs to fp32
  std::vector<Variable> casted_inputs;
  for (const auto& var : instr->inputs) {
    auto casted_var = var;
    if (var->type.is_float16() || var->type.is_bfloat16()) {
      casted_var = builder->Cast(var, "float32");
    }
    casted_inputs.emplace_back(casted_var);
  }
  // Run fp32 op
  const auto& outputs =
      builder->CustomInstr(instr->op_type, casted_inputs, instr->attrs);
  // Cast all fp32 outputs to fp16/bf16
  for (int i = 0; i < outputs.size(); ++i) {
    if (outputs[i]->type.is_float(32)) {
      builder->AppendInstruction(
          CreateNewCastInstruction(outputs[i], instr->outputs[i]));
    }
  }
}

static std::unordered_map<std::string, CastImplFunc> need_cast_list = {
    // math function
    {"sin", CommonCastImpl},
    {"cos", CommonCastImpl},
    {"exp", CommonCastImpl},
    {"log", CommonCastImpl},
    {"log2", CommonCastImpl},
    {"log10", CommonCastImpl},
    {"sqrt", CommonCastImpl},
    {"rsqrt", CommonCastImpl},
    {"cbrt", CommonCastImpl},
    {"erf", CommonCastImpl},
    {"sinh", CommonCastImpl},
    {"cosh", CommonCastImpl},
    {"tanh", CommonCastImpl},
    {"asin", CommonCastImpl},
    {"acos", CommonCastImpl},
    {"atan", CommonCastImpl},
    {"asinh", CommonCastImpl},
    {"acosh", CommonCastImpl},
    {"atanh", CommonCastImpl},
    {"remainder", CommonCastImpl},
    {"pow", CommonCastImpl},
    // reduce
    {"reduce_sum", CommonCastImpl},
    {"reduce_prod", CommonCastImpl},
    // composite function
    {"sigmoid", CommonCastImpl},
    {"sum", CommonCastImpl},
    {"softmax", CommonCastImpl},
    {"gelu", CommonCastImpl},
    {"batch_norm",
     [](NetBuilder* builder, const Instruction& instr) {
       if (!IsInputHasFP16OrBF16(instr->inputs)) {
         // DO NOT NEED CAST
         builder->AppendInstruction(instr);
         return;
       }

       // Except input [X], BatchNormTrain's Input should all be fp32
       CHECK_EQ(instr->inputs.size(), 5UL)
           << "The number of the given inputs is not equal to the required for "
              "op "
           << instr->op_type;
       CHECK(instr->inputs[1]->type.is_float(32))
           << instr->op_type << "'s input [scale] should be float32, but here "
           << instr->inputs[1]->type;
       CHECK(instr->inputs[2]->type.is_float(32))
           << instr->op_type << "'s input [bias] should be float32, but here "
           << instr->inputs[1]->type;
       CHECK(instr->inputs[3]->type.is_float(32))
           << instr->op_type
           << "'s input [moving_mean] should be float32, but here "
           << instr->inputs[1]->type;
       CHECK(instr->inputs[4]->type.is_float(32))
           << instr->op_type
           << "'s input [moving_variance] should be float32, but here "
           << instr->inputs[1]->type;

       // Cast input [X] from fp16/bf16 to fp32
       const auto& x = instr->inputs[0];
       const auto& x_casted = builder->Cast(x, "float32");

       auto casted_inputs = instr->inputs;
       casted_inputs[0] = x_casted;
       // Run fp32 function
       const auto& outputs =
           builder->CustomInstr(instr->op_type, casted_inputs, instr->attrs);
       // Cast output [Y] to fp16/bf16, no other output
       builder->AppendInstruction(
           CreateNewCastInstruction(outputs[0], instr->outputs[0]));
     }},
    {"batch_norm_train",
     [](NetBuilder* builder, const Instruction& instr) {
       if (!IsInputHasFP16OrBF16(instr->inputs)) {
         // DO NOT NEED CAST
         builder->AppendInstruction(instr);
         return;
       }

       // Except input [X], BatchNormTrain's Input should all be fp32
       CHECK_EQ(instr->inputs.size(), 5UL)
           << "The number of the given inputs is not equal to the required for "
              "op "
           << instr->op_type;
       CHECK(instr->inputs[1]->type.is_float(32))
           << instr->op_type << "'s input [scale] should be float32, but here "
           << instr->inputs[1]->type;
       CHECK(instr->inputs[2]->type.is_float(32))
           << instr->op_type << "'s input [bias] should be float32, but here "
           << instr->inputs[1]->type;
       CHECK(instr->inputs[3]->type.is_float(32))
           << instr->op_type
           << "'s input [moving_mean] should be float32, but here "
           << instr->inputs[1]->type;
       CHECK(instr->inputs[4]->type.is_float(32))
           << instr->op_type
           << "'s input [moving_variance] should be float32, but here "
           << instr->inputs[1]->type;

       // Cast input [X] from fp16/bf16 to fp32
       const auto& x = instr->inputs[0];
       const auto& x_casted = builder->Cast(x, "float32");

       auto casted_inputs = instr->inputs;
       casted_inputs[0] = x_casted;
       // Run fp32 function
       const auto& outputs =
           builder->CustomInstr(instr->op_type, casted_inputs, instr->attrs);
       // Cast output [Y] to fp16/bf16
       builder->AppendInstruction(
           CreateNewCastInstruction(outputs[0], instr->outputs[0]));
       // Identity other output
       for (int i = 1; i < outputs.size(); ++i) {
         builder->AppendInstruction(
             CreateNewIdentityInstruction(outputs[i], instr->outputs[i]));
       }
     }},
    {"batch_norm_grad", [](NetBuilder* builder, const Instruction& instr) {
       if (!IsInputHasFP16OrBF16(instr->inputs)) {
         // DO NOT NEED CAST
         builder->AppendInstruction(instr);
         return;
       }

       // Except input [X], BatchNormTrain's Input should all be fp32
       CHECK_EQ(instr->inputs.size(), 5UL)
           << "The number of the given inputs is not equal to the required for "
              "op "
           << instr->op_type;
       CHECK_EQ(instr->inputs[0]->type, instr->inputs[1]->type)
           << instr->op_type
           << "'s input [Y@GRAD] and input [X] 's type should be the same";
       CHECK(instr->inputs[2]->type.is_float(32))
           << instr->op_type << "'s input [scale] should be float32, but here "
           << instr->inputs[1]->type;
       CHECK(instr->inputs[3]->type.is_float(32))
           << instr->op_type
           << "'s input [save_mean] should be float32, but here "
           << instr->inputs[1]->type;
       CHECK(instr->inputs[4]->type.is_float(32))
           << instr->op_type
           << "'s input [save_variance] should be float32, but here "
           << instr->inputs[1]->type;

       // Cast input [Y@GRAD] from fp16/bf16 to fp32
       const auto& y_grad = instr->inputs[0];
       const auto& y_grad_casted = builder->Cast(y_grad, "float32");

       // Cast input [X] from fp16/bf16 to fp32
       const auto& x = instr->inputs[1];
       const auto& x_casted = builder->Cast(x, "float32");

       auto casted_inputs = instr->inputs;
       casted_inputs[0] = y_grad_casted;
       casted_inputs[1] = x_casted;
       // Run fp32 function
       const auto& outputs =
           builder->CustomInstr(instr->op_type, casted_inputs, instr->attrs);
       // Cast output [X@GRAD] to fp16/bf16
       builder->AppendInstruction(
           CreateNewCastInstruction(outputs[0], instr->outputs[0]));
       // Identity other output
       for (int i = 1; i < outputs.size(); ++i) {
         builder->AppendInstruction(
             CreateNewIdentityInstruction(outputs[i], instr->outputs[i]));
       }
     }}};
}  // namespace

class AutoCastPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 protected:
  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    NetBuilder builder("auto_cast_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }
    for (int i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];

      if (need_cast_list.count(instr->op_type)) {
        need_cast_list.at(instr->op_type)(&builder, instr);
      } else {
        builder.AppendInstruction(instr);
      }
    }
    *program = builder.Build();
  }

  void Clear() override {}
};

}  // namespace pass
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(AutoCast) {
  CINN_REGISTER_PROGRAM_PASS(AutoCast, cinn::frontend::pass::AutoCastPass);

  return true;
}
