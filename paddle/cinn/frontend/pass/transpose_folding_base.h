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

#pragma once

#include <absl/container/flat_hash_map.h>

#include <sstream>
#include <string>
#include <unordered_set>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/program_pass.h"
#include "paddle/cinn/frontend/syntax.h"

namespace cinn::frontend::pass {

class TransposeFoldingBase : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;
  using In2InstrType =
      absl::flat_hash_map<std::string, std::unordered_set<Instruction*>>;
  using Out2InstrType = absl::flat_hash_map<std::string, Instruction*>;

 protected:
  virtual void set_target_instrs() = 0;
  // the ops which can folding into matmul
  void set_fold_instrs() {
    fold_instrs_ = {"transpose", "scale", "broadcast_to"};
  }
  // the ops which cannot folding but can ignore when it place between into
  // folding op and matmul
  void set_skip_instrs() { skip_instrs_ = {"cast", "identity"}; }

  void Clear() override {
    target_instrs_.clear();
    fold_instrs_.clear();
    skip_instrs_.clear();
  }

  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    set_target_instrs();
    set_fold_instrs();
    set_skip_instrs();
    // `out2instr` is used to represent the mapping of Output to Instruction.
    Out2InstrType out2instr;
    // `in2instr` is used to represent the mapping of Input to Instruction.
    In2InstrType in2instr;
    for (size_t i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];
      for (const auto& out : instr->outputs) {
        out2instr[out->id] = &instr;
      }
      for (const auto& in : instr->inputs) {
        in2instr[in->id].insert(&instr);
      }
    }

    // `remove_instrs` is used to represent Instructions of which type is
    // transpose to be deleted.
    absl::flat_hash_set<Instruction*> remove_instrs;
    for (size_t i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];
      if (target_instrs_.count(instr->op_type)) {
        DoMatmulFoldOptimize(
            &instr, out2instr, in2instr, fetch_ids, &remove_instrs);
      }
    }

    NetBuilder builder("transpose_folding_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }
    for (int i = 0; i < program->size(); i++) {
      if (!remove_instrs.count(&(*program)[i])) {
        builder.AppendInstruction((*program)[i]);
      }
    }
    *program = builder.Build();
  }

  // get can fold instruction in order, more front, more near from dot op
  // the `instr` param is the next instruction of matmul, not the matmul
  std::vector<Instruction*> GetFoldInstruction(Instruction* instr,
                                               const Out2InstrType& out2instr,
                                               const In2InstrType& in2instr,
                                               bool from_input) const {
    if (!fold_instrs_.count((*instr)->op_type) &&
        !skip_instrs_.count((*instr)->op_type)) {
      return {};
    }
    CHECK_EQ((*instr)->inputs.size(), 1UL)
        << "The op " << (*instr)->op_type << " should has 1 input.";
    CHECK_EQ((*instr)->outputs.size(), 1UL)
        << "The op " << (*instr)->op_type << " should has 1 output.";

    VLOG(5) << "Try get matmul's folding instructions begin from ["
            << (*instr)->inputs[0]->id << "]";

    if (!from_input && in2instr.at((*instr)->inputs[0]->id).size() != 1UL) {
      // the matmul's output should only link to one op
      VLOG(5) << "The var [" << (*instr)->inputs[0]->id
              << "] link to many op, cannot fold into matmul! Please check.";
      return {};
    }

    std::vector<Instruction*> res = {instr};
    std::unordered_set<std::string> visited = {(*instr)->op_type};

    auto cur_instr = instr;
    while (cur_instr) {
      Instruction* next_instr = nullptr;

      if (from_input) {
        // scale -> transpose -> matmul ==> {"transpose", "scale"}
        auto iter = out2instr.find((*cur_instr)->inputs[0]->id);
        if (iter != out2instr.end()) {
          next_instr = iter->second;
        }
      } else {
        // matmul -> transpose -> scale ==> {"transpose", "scale"}
        auto iter = in2instr.find((*cur_instr)->outputs[0]->id);
        if (iter != in2instr.end() && iter->second.size() == 1UL) {
          next_instr = *iter->second.begin();
        }
      }

      if (CanFold(next_instr, visited)) {
        // found can fold instruction and not repeat
        res.emplace_back(next_instr);
        visited.emplace((*next_instr)->op_type);
      } else {
        // the fold instructions must consecutive
        break;
      }

      cur_instr = next_instr;
    }

    return res;
  }

  bool CanFold(const Instruction* instr,
               const std::unordered_set<std::string>& visited_instr) const {
    if (!instr) {
      return false;
    }

    const auto& instr_type = (*instr)->op_type;
    if ((!fold_instrs_.count(instr_type) && !skip_instrs_.count(instr_type)) ||
        visited_instr.count(instr_type)) {
      return false;
    }
    if (instr_type == "transpose") {
      if (visited_instr.count("broadcast_to")) {
        // if transpose after broadcast_to, cannot fold because shape has
        // changed
        return false;
      }
    }
    return true;
  }

  bool IsValidTranspose(const Instruction& transpose) const {
    if ("transpose" != transpose->op_type) {
      return false;
    }

    // `axis` of tranpose must be consecutive in the reverse order,
    // excluding the first dim
    auto axis = transpose.GetAttrs<std::vector<int>>("axis");
    if (axis.size() <= 1) {
      return false;
    }
    int rank = axis.size();

    // the batch dimension should not change
    for (int i = 0; i < rank - 2; ++i) {
      if (axis[i] != i) {
        return false;
      }
    }
    // only the last two dimension need transpose
    if (axis[rank - 2] != rank - 1 || axis[rank - 1] != rank - 2) {
      return false;
    }

    return true;
  }

  bool IsValidScale(const Instruction& scale) const {
    if ("scale" != scale->op_type) {
      return false;
    }

    float bias = scale->attrs.count("bias")
                     ? absl::get<float>(scale->attrs.at("bias"))
                     : 0.0f;
    return (bias == 0.0f);
  }

  bool CanSkip(const Instruction& instr) const {
    return skip_instrs_.count(instr->op_type);
  }

  virtual void DoMatmulFoldOptimize(
      Instruction* instr,
      const Out2InstrType& out2instr,
      const In2InstrType& in2instr,
      const std::unordered_set<std::string>& fetch_ids,
      absl::flat_hash_set<Instruction*>* remove_instrs) const = 0;

  std::unordered_set<std::string> target_instrs_;
  std::unordered_set<std::string> fold_instrs_;
  std::unordered_set<std::string> skip_instrs_;
};

}  // namespace cinn::frontend::pass
