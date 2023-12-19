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

#include <absl/container/flat_hash_map.h>
#include <absl/types/variant.h>

#include <string>
#include <unordered_set>

#include "paddle/cinn/frontend/pass/transpose_folding_base.h"
#include "paddle/cinn/frontend/program_pass.h"
#include "paddle/cinn/frontend/syntax.h"

namespace cinn::frontend::pass {

// Pass `TransposeFoldingInput` folds transpose into dot, then both of them can
// be implemented by a GEMM kernel. For each dot operator, try folding every
// input that belong output of transpose. If output of tranpose in `fetch_ids`,
// keep the operator.
class TransposeFoldingInputPass : public TransposeFoldingBase {
 public:
  using TransposeFoldingBase::TransposeFoldingBase;

 protected:
  void set_target_instrs() override {
    TransposeFoldingBase::target_instrs_ = {"matmul"};
  }

  bool IsValidBroadCast(const Instruction& broadcast,
                        const Instruction& dot,
                        const int input_id) const {
    if ("broadcast_to" != broadcast->op_type) {
      return false;
    }

    // check whether the output shape can infer from another input, if not,
    // cannot remove this broadcast
    int next_id = (input_id + 1) % dot->inputs.size();
    const auto& next_shape = dot->inputs[next_id]->shape;
    const auto& out_shape = dot->outputs[0]->shape;

    if (next_shape.size() != out_shape.size()) {
      return false;
    }

    for (int i = 0; i < next_shape.size() - 2; ++i) {
      if (next_shape[i] != out_shape[i]) {
        return false;
      }
    }
    return true;
  }

  void DoMatmulFoldOptimize(
      Instruction* dot,
      const Out2InstrType& out2instr,
      const In2InstrType& in2instr,
      const std::unordered_set<std::string>& fetch_ids,
      absl::flat_hash_set<Instruction*>* remove_instrs) const override {
    CHECK_EQ((*dot)->inputs.size(), 2UL)
        << "The matmul should only have two inputs.";

    auto debug_info = [](const std::vector<Instruction*>& instrs) {
      std::stringstream ss;
      for (auto instr : instrs) {
        ss << (*instr)->op_type << ", ";
      }
      return ss.str();
    };

    for (size_t i = 0; i < (*dot)->inputs.size(); ++i) {
      auto iter = out2instr.find((*dot)->inputs[i]->id);
      if (iter != out2instr.end()) {
        // for example: x -> scale -> y -> transpose -> z -> dot
        // fold_instrs = {"transpose", "scale"}
        const auto& fold_instrs =
            GetFoldInstruction(iter->second, out2instr, in2instr, true);

        if (fold_instrs.empty()) {
          continue;
        }

        VLOG(4) << "Fold Instruction: [" << debug_info(fold_instrs) << "]"
                << " into " << (i == 0 ? "x" : "y") << " of matmul: " << *dot;

        bool shape_has_changed = false;
        for (int j = fold_instrs.size() - 1; j >= 0; --j) {
          auto instr = fold_instrs[j];

          if (IsValidTranspose(*instr)) {
            // fold transpose into trans_a/trans_b
            if (i == 0) {
              bool trans_a = (*dot)->attrs.count("trans_a")
                                 ? absl::get<bool>((*dot)->attrs.at("trans_a"))
                                 : false;
              dot->SetAttr("trans_a", static_cast<bool>(trans_a ^ true));
            } else if (i == 1) {
              bool trans_b = (*dot)->attrs.count("trans_b")
                                 ? absl::get<bool>((*dot)->attrs.at("trans_b"))
                                 : false;
              dot->SetAttr("trans_b", static_cast<bool>(trans_b ^ true));
            } else {
              LOG(FATAL) << "The matmul should only have two inputs.";
            }

            // shape has changed, the ignore op should update shape
            shape_has_changed = true;
          } else if (IsValidScale(*instr)) {
            // assume C = alpha * A * B + beta * C
            // fold scale into alpha
            float scale = (*instr)->attrs.count("scale")
                              ? absl::get<float>((*instr)->attrs.at("scale"))
                              : 1.0f;

            float alpha = (*dot)->attrs.count("alpha")
                              ? absl::get<float>((*dot)->attrs.at("alpha"))
                              : 1.0f;
            dot->SetAttr("alpha", alpha * scale);
          } else if (IsValidBroadCast(*instr, *dot, i)) {
            // nothin to do, can fold directly
            // the x's broadcast has removed, cannot removed the y's
            shape_has_changed = true;
          } else if (CanSkip(*instr)) {
            if (shape_has_changed) {
              // the transpose op may change the shape, need update
              (*instr)->outputs[0]->shape = (*instr)->inputs[0]->shape;
            }
            continue;
          } else {
            // invlid folding op, skip
            continue;
          }

          // relink input: x-> transpose -> out -> dot ==> x -> dot
          auto next_instr = (j == 0) ? dot : fold_instrs[j - 1];
          if (j == 0) {
            (*dot)->inputs[i] = (*instr)->inputs[0];
          } else {
            (*next_instr)->inputs[0] = (*instr)->inputs[0];
          }

          // check whether the instruction can be removed
          const auto& out_name = (*instr)->outputs[0]->id;
          const auto& out_instrs = in2instr.at(out_name);

          bool can_remove = std::all_of(
              out_instrs.begin(),
              out_instrs.end(),
              [&](Instruction* out_instr) {
                // the transpose had linked to not matmul op, cannot remove
                return target_instrs_.count((*out_instr)->op_type) ||
                       (out_instr == next_instr);
              });

          if (can_remove && !fetch_ids.count(out_name)) {
            // the transpose is only link to matmul and its output is not in
            // fetch_ids, should remove
            remove_instrs->insert(instr);
          }
        }
      }
    }
  }
};

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(TransposeFoldingInput) {
  CINN_REGISTER_PROGRAM_PASS(TransposeFoldingInput,
                             ::cinn::frontend::pass::TransposeFoldingInputPass);

  return true;
}
