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

class ExpandZeroDimPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 protected:
  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    NetBuilder builder("expand_zero_dim_builder");
    for (int i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];
      if (instr->op_type == "transpose") {
        builder.AppendInstruction(HandleTranspose(instr));
        continue;
      } else if (instr->op_type == "fill_constant") {
        builder.AppendInstruction(HandleFillConstant(instr));
        continue;
      }
      for (auto& input : instr->inputs) {
        if (input->shape.empty()) {
          VLOG(4) << "Change " << instr->op_type << "'s input 0D-Tensor "
                  << input->id << " to 1D-Tensor";
          input->shape.push_back(1);
        }
      }
      for (auto& output : instr->outputs) {
        if (output->shape.empty()) {
          VLOG(4) << "Change " << instr->op_type << "'s output 0D-Tensor "
                  << output->id << " to 1D-Tensor";
          output->shape.push_back(1);
        }
      }
      builder.AppendInstruction(instr);
    }
    for (auto var : program->GetInputs()) {
      if (var->shape.empty()) {
        VLOG(4) << "Change program's input 0D-Tensor " << var->id
                << " to 1D-Tensor";
        var->shape.push_back(1);
      }
      builder.CreateInput(var);
    }
    *program = builder.Build();
  }

  void Clear() override {}

 private:
  // Before: out-0D = transpose(x-0D, [])
  // After:  out-1D = transpose(x-1D, [1])
  Instruction HandleTranspose(const Instruction& instr) {
    Instruction new_instr = instr;
    bool has_0d_input = false;
    for (auto& input : new_instr->inputs) {
      if (input->shape.empty()) {
        VLOG(4) << "Change transpose's input 0D-Tensor " << input->id
                << " to 1D-Tensor";
        input->shape.push_back(1);
        has_0d_input = true;
      }
    }
    for (auto& output : new_instr->outputs) {
      if (output->shape.empty()) {
        VLOG(4) << "Change transpose's output 0D-Tensor " << output->id
                << " to 1D-Tensor";
        output->shape.push_back(1);
      }
    }
    if (has_0d_input) {
      std::vector<int32_t> axis =
          new_instr.GetAttrs<std::vector<int32_t>>("axis");
      CHECK(axis.empty()) << "transpose's axis should be empty when inputs "
                             "0D-Tensor! Please check setting.\n";
      axis.push_back(0);
      VLOG(4) << "Change Transpose's attribute axis from [] to [1]";
      new_instr.SetAttr<std::vector<int32_t>>("axis", axis);
    }
    return new_instr;
  }

  // Before: out-0D = fill_constant([], 123.456, "out", "float32")
  // After:  out-1D = fill_constant([1], 123.456, "out", "float32")
  Instruction HandleFillConstant(const Instruction& instr) {
    Instruction new_instr = instr;
    std::vector<int32_t> shape =
        new_instr.GetAttrs<std::vector<int32_t>>("shape");
    if (shape.empty()) {
      shape.push_back(1);
      VLOG(4) << "Change fill_constant's attribute shape from [] to [1]";
    }
    new_instr.SetAttr<std::vector<int32_t>>("shape", shape);
    return new_instr;
  }
};

}  // namespace pass
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(ExpandZeroDim) {
  CINN_REGISTER_PROGRAM_PASS(ExpandZeroDim,
                             cinn::frontend::pass::ExpandZeroDimPass);

  return true;
}
