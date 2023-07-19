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
#include <vector>

#include "glog/logging.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/program_pass.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/cinn/utils/type_defs.h"

namespace cinn {
namespace frontend {
namespace pass {

class AutoBroadcastPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 private:
  std::vector<int> GetBroadcastAxes(const cinn::utils::ShapeType& input_shape,
                                    const cinn::utils::ShapeType& output_shape,
                                    int axis) {
    std::vector<int> broadcast_axes;
    if (input_shape.size() == output_shape.size()) {
      for (int idx = 0; idx < input_shape.size(); ++idx) {
        broadcast_axes.push_back(idx);
      }
    } else {
      if (axis == -1) {
        axis = output_shape.size() - input_shape.size();
      }
      CHECK_LE(axis + input_shape.size(), output_shape.size())
          << "Cannot Broadcast from shape=["
          << cinn::utils::Join(input_shape, ", ") << "] to shape=["
          << cinn::utils::Join(output_shape, ", ") << "] with axis=" << axis;
      for (int idx = 0; idx < input_shape.size(); ++idx) {
        broadcast_axes.push_back(axis++);
      }
    }
    return broadcast_axes;
  }

  void InsertBroadcastTo(NetBuilder* builder, Instruction* broadcast_op) {
    const auto& instr = *broadcast_op;
    const auto& op_name = instr->op_type;

    const auto& op_pattern_dict_ = &cinn::hlir::framework::Operator::GetAttrs<
        cinn::hlir::framework::OpPatternKind>("OpPattern");
    const auto* op = cinn::hlir::framework::Operator::Get(op_name);
    if (!op_pattern_dict_->Find(op) ||
        (*op_pattern_dict_)[op] != cinn::hlir::framework::kBroadcast) {
      // no set OpPattern or not broadcast kind operator, skip
      builder->AppendInstruction(instr);
      return;
    }
    if (instr->inputs.size() <= 1) {
      // skip broadcast_to and other op
      builder->AppendInstruction(instr);
      return;
    }

    const auto& outputs = instr.GetOutputs();
    CHECK_EQ(outputs.size(), 1)
        << "The broadcast operator should has and only has one output";
    const auto& output = outputs.front();

    int axis = -1;
    if (instr->attrs.count("axis")) {
      axis = instr.GetAttrs<int>("axis");
    }

    bool need_insert = false;
    std::vector<Variable> new_inputs;
    for (auto input : instr->inputs) {
      if (input->shape == output->shape) {
        // if shape same, no need broadcast
        new_inputs.emplace_back(input);
      } else {
        // else insert broadcast_to
        need_insert = true;

        auto new_var = builder->BroadcastTo(
            input,
            output->shape,
            GetBroadcastAxes(input->shape, output->shape, axis));
        new_inputs.emplace_back(new_var);
      }
    }

    if (need_insert) {
      VLOG(4) << "Before Insert broadcast_to: " << *broadcast_op;
      // update origin broadcast op's input and attribute
      broadcast_op->SetInputs(std::move(new_inputs));
      (*broadcast_op)->attrs["axis"] = -1;
      VLOG(4) << "After Insert broadcast_to: " << *broadcast_op;
    }
    // append new broadcast
    builder->AppendInstruction(*broadcast_op);
  }

 protected:
  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    NetBuilder builder("auto_broadcast_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }
    for (int i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];

      InsertBroadcastTo(&builder, &instr);
    }
    *program = builder.Build();
  }

  void Clear() override {}
};

}  // namespace pass
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(AutoBroadcast) {
  CINN_REGISTER_PROGRAM_PASS(AutoBroadcast,
                             cinn::frontend::pass::AutoBroadcastPass);

  return true;
}
