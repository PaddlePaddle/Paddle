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

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/program_pass.h"
#include "paddle/cinn/frontend/syntax.h"

namespace cinn::frontend::pass {

using cinn::utils::DimType;
using cinn::utils::ShapeType;

class TransposeKey {
 public:
  TransposeKey(const std::string& input_id, const ShapeType& axis) {
    SetKey(input_id, axis);
  }

  void SetKey(const std::string& input_id, const ShapeType& axis) {
    input_id_ = input_id;
    axis_ = axis;
  }

  bool operator==(const TransposeKey& other) const {
    return axis_ == other.axis_ && input_id_ == other.input_id_;
  }
  bool operator!=(const TransposeKey& other) const {
    return !this->operator==(other);
  }

  struct Hash {
    size_t operator()(const TransposeKey& key) const {
      std::string ret;

      ret.append(key.input_id_);
      std::for_each(
          key.axis_.begin(), key.axis_.end(), [&](const DimType& dim) {
            ret.append(std::to_string(dim));
          });

      return std::hash<std::string>()(ret);
    }
  };

 private:
  std::string input_id_;
  ShapeType axis_;
};

// Pass `TransposeCollapsing` folds multi transpose into one.
class TransposeCollapsingPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;
  using OutputToOpMap = std::unordered_map<std::string, Instruction*>;
  using InputToOpMap =
      std::unordered_map<std::string, std::unordered_set<Instruction*>>;

 protected:
  void Clear() override {}

  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) const override {
    // `out2instr` is used to represent the mapping of Output to Instruction.
    OutputToOpMap out2instr;
    // `in2instr` is used to represent the mapping of Input to Instruction.
    InputToOpMap in2instr;
    // all transpose op in program
    std::unordered_set<Instruction*> all_transpose;

    for (size_t i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];
      for (const auto& out : instr->outputs) {
        out2instr[out->id] = &instr;
      }
      for (const auto& in : instr->inputs) {
        in2instr[in->id].insert(&instr);
      }
      if ("transpose" == instr->op_type) {
        all_transpose.insert(&instr);
      }
    }

    // the useless transpose op need to remove from program
    std::unordered_set<Instruction*> remove_instrs;
    FoldingTransposeVertical(
        all_transpose, fetch_ids, in2instr, out2instr, &remove_instrs);

    for (auto instr : remove_instrs) {
      if (all_transpose.count(instr)) {
        all_transpose.erase(instr);
      }
    }
    // TODO(thisjiang): reopen after CINN support recompute for performance
    // due to recompute unsupported, if the op output to two group, it will also
    // create a new group, so that the horizontal fuse will not improve
    // performance. FoldingTransposeHorizontal(all_transpose, fetch_ids,
    // in2instr, out2instr, &remove_instrs);

    NetBuilder builder("transpose_collapsing_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }
    for (int i = 0; i < program->size(); i++) {
      if (remove_instrs.end() == remove_instrs.find(&(*program)[i])) {
        builder.AppendInstruction((*program)[i]);
      }
    }
    *program = builder.Build();
  }

 private:
  void FoldingTransposeVertical(
      const std::unordered_set<Instruction*>& all_transpose,
      const std::unordered_set<std::string>& fetch_ids,
      const InputToOpMap& in2instr,
      const OutputToOpMap& out2instr,
      std::unordered_set<Instruction*>* remove_instrs) const {
    if (all_transpose.size() == 1) {
      return;
    }
    // the transpose op should not remove
    std::unordered_set<Instruction*> visited_instrs;
    for (auto transpose : all_transpose) {
      if (("transpose" != (*transpose)->op_type) ||
          visited_instrs.count(transpose)) {
        // the transpose op had been fused, skip
        continue;
      }

      // Fuse transpose from front to back, the fuse path is unique
      auto first_transpose = FindFirstTranspose(transpose, out2instr);
      TryFuseTranspose(
          first_transpose, fetch_ids, in2instr, remove_instrs, &visited_instrs);
    }
  }

  Instruction* FindFirstTranspose(Instruction* transpose,
                                  const OutputToOpMap& out2instr) const {
    auto first_transpose = transpose;

    auto input_name = (*first_transpose)->inputs.front()->id;
    // Q: Why check whether transpose's input in out2instr ?
    // A: The input may be the input of the graph other than another op's
    // output.
    //    Obviously, the transpose op is the first transpose in the situation.
    while (out2instr.count(input_name)) {
      auto instr = out2instr.at(input_name);
      if ("transpose" != (*instr)->op_type) {
        // if input of transpose is not output of another transpose, it is the
        // first transpose.
        break;
      }

      input_name = (*instr)->inputs.front()->id;
      first_transpose = instr;
    }
    return first_transpose;
  }

  void TryFuseTranspose(
      Instruction* transpose,
      const std::unordered_set<std::string>& fetch_ids,
      const InputToOpMap& in2instr,
      std::unordered_set<Instruction*>* remove_instrs,
      std::unordered_set<Instruction*>* visited_instrs) const {
    visited_instrs->insert(transpose);

    const auto& input = (*transpose)->inputs.front();
    const auto& input_name = input->id;

    const auto& output = (*transpose)->outputs.front();
    const auto& output_name = output->id;

    const auto& axis = transpose->GetAttrs<ShapeType>("axis");
    CHECK_EQ(axis.size(), input->shape.size())
        << "The transpose's axis size should equal with input variable's shape "
           "size, but the transpose of ["
        << input->id << "] not ! Please check.";

    bool can_remove = !fetch_ids.count(output_name);

    if (CheckTransposeBorder(transpose, in2instr)) {
      if (can_remove) {
        VLOG(4) << "The transpose op {input[" << input_name << "], output["
                << output_name << "], axis[" << cinn::utils::Join(axis, ",")
                << "]} is a output op of graph, connot fuse, remove.";
        // this transpose not used by any other op, remove
        remove_instrs->insert(transpose);
      } else {
        if (CheckTransposeUseless(axis)) {
          VLOG(4) << "The transpose op {input[" << input_name << "], output["
                  << output_name << "], axis[" << cinn::utils::Join(axis, ",")
                  << "]} is fetched but useless, replace with identity.";
          // cannot remove, however, the transpsoe is useless, we can replace
          // the transpose with indentiy for more fusion opportunity
          ReplaceWithIdentity(transpose);
        }
        // else the transpsoe is fetched and helpful, ignore
      }
      return;
    }

    // CheckTransposeBorder ensure `output_name` existed in `in2instr`
    const auto& out_instrs = in2instr.at(output_name);
    if (CheckTransposeUseless(axis)) {
      if (!can_remove) {
        VLOG(4) << "The transpose op {input[" << input_name << "], output["
                << output_name << "], axis[" << cinn::utils::Join(axis, ",")
                << "]} is useless but fetched, replace with identity.";
        // cannot remove, but we can replace the transpose with indentiy for
        // more fusion opportunity
        ReplaceWithIdentity(transpose);
      } else {
        VLOG(4) << "The transpose op {input[" << input_name << "], output["
                << output_name << "], axis[" << cinn::utils::Join(axis, ",")
                << "]} is useless, remove.";
        for (auto instr : out_instrs) {
          // replace the input to transpose's input
          ReplaceInputVariable(instr, output_name, input);
        }
        remove_instrs->insert(transpose);

        for (auto instr : out_instrs) {
          if ("transpose" == (*instr)->op_type) {
            // if the next instruction is transpose op, continue fuse
            TryFuseTranspose(
                instr, fetch_ids, in2instr, remove_instrs, visited_instrs);
          }
        }
      }
      return;
    }

    if (!CheckOutputContainTranspose(transpose, in2instr)) {
      VLOG(4) << "The transpose op {input[" << input_name << "], output["
              << output_name << "], axis[" << cinn::utils::Join(axis, ",")
              << "]} doesn't has output link to transpose, skip.";
      return;
    }

    std::unordered_set<Instruction*> next_fused_instrs;

    for (auto instr : out_instrs) {
      if ("transpose" != (*instr)->op_type) {
        // the transpose was used by other non-transpose op, cannot remove, skip
        can_remove = false;
        VLOG(4) << "Fuse transpose of {input[" << input_name << "], output["
                << output_name << "], axis [" << cinn::utils::Join(axis, ",")
                << "]} was used by " << (*instr)->op_type << ", cannot remove.";
        continue;
      }

      const auto& next_axis = instr->GetAttrs<ShapeType>("axis");
      // we can fuse two transpose by fuse the two axes like:
      // step |    axis   | after_transpose
      //  1   | [0, 2, 1] | [0, 2, 1]
      //  2   | [2, 1, 0] | [1, 2, 0]
      // so we can fuse tranpose([0, 2, 1]) and tranpose([2, 1, 0]) into
      // tranpose([1, 2, 0])
      const auto& fused_axis = FuseTransposeAxis(axis, next_axis);

      VLOG(4) << "Fuse transpose of {input[" << input_name << "], output["
              << output_name << "], axis [" << cinn::utils::Join(axis, ",")
              << "]} and transpose of {input[" << (*instr)->inputs.front()->id
              << "], output[" << (*instr)->outputs.front()->id << "], axis ["
              << cinn::utils::Join(next_axis, ",")
              << "]} into transpose of {input[" << input_name << "], output["
              << (*instr)->outputs.front()->id << "], axis["
              << cinn::utils::Join(fused_axis, ",") << "]}.";

      auto fused_transpose = FuseTransposeImpl(transpose, instr, fused_axis);

      next_fused_instrs.insert(fused_transpose);
    }

    if (can_remove) {
      VLOG(4) << "Remove transpose of {input[" << input_name << "], output["
              << output_name << "], axis [" << cinn::utils::Join(axis, ",")
              << "]}.";
      remove_instrs->insert(transpose);
    }

    for (auto instr : next_fused_instrs) {
      TryFuseTranspose(
          instr, fetch_ids, in2instr, remove_instrs, visited_instrs);
    }
  }

  // check whether the op is the border op of graph, in other words, its output
  // var was not used by any op in graph.
  bool CheckTransposeBorder(Instruction* transpose,
                            const InputToOpMap& in2instr) const {
    const auto& output_name = (*transpose)->outputs.front()->id;
    return !in2instr.count(output_name) || in2instr.at(output_name).empty();
  }

  // check whether the op's output ops has transpose, if not, no transpose need
  // folding
  bool CheckOutputContainTranspose(Instruction* transpose,
                                   const InputToOpMap& in2instr) const {
    const auto& output_name = (*transpose)->outputs.front()->id;
    for (auto instr : in2instr.at(output_name)) {
      if ("transpose" == (*instr)->op_type) {
        return true;
      }
    }
    // the first transpose's output is not anyone transpose's input
    return false;
  }

  // if the transpose axis like {0, 1, 2, 3, 4, 5}, the transpose is useless,
  // should remove
  bool CheckTransposeUseless(const ShapeType& axis) const {
    for (int i = 0; i < axis.size(); ++i) {
      if (axis[i] != i) {
        return false;
      }
    }
    return true;
  }

  // replace the op's input variable whose name is `old_input_name` to
  // `new_input`, note we need keep the input list order
  void ReplaceInputVariable(Instruction* op,
                            const std::string& old_input_name,
                            const Variable& new_input) const {
    auto find_input = [&](const std::string& input_name) {
      return std::find_if(
          (*op)->inputs.begin(), (*op)->inputs.end(), [&](const Variable& v) {
            return input_name == v->id;
          });
    };

    // Why Loop : To avoid the op's inputs are the same variable !
    for (auto it = find_input(old_input_name); it != (*op)->inputs.end();
         it = find_input(old_input_name)) {
      // erase previous fill_constant output var and replace to new
      // fill_constant output var
      auto next_it = (*op)->inputs.erase(it);
      // keep the input place same, it's very important
      (*op)->inputs.insert(next_it, new_input);
    }
  }

  Instruction* ReplaceWithIdentity(Instruction* op) const {
    (*op)->op_type = "identity";
    (*op)->attrs.clear();
    (*op)->attrs_ordered.clear();
    return op;
  }

  // compute the fused axis of `old_axis` and `new_axis`, like [0, 2, 1] + [2,
  // 1, 0] = [1, 2, 0]
  ShapeType FuseTransposeAxis(const ShapeType& old_axis,
                              const ShapeType& new_axis) const {
    CHECK_EQ(old_axis.size(), new_axis.size())
        << "The transpose axis size should be " << old_axis.size()
        << ", but here " << new_axis.size();

    ShapeType axis = old_axis;
    for (int i = 0; i < new_axis.size(); ++i) {
      axis[i] = old_axis[new_axis[i]];
    }
    return axis;
  }

  // fuse the two transpose axis into the second transpose, replace its input
  // and axis
  Instruction* FuseTransposeImpl(Instruction* transpose1,
                                 Instruction* transpose2,
                                 const ShapeType& fused_axis) const {
    (*transpose2)->inputs.front() = (*transpose1)->inputs.front();
    transpose2->SetAttr("axis", fused_axis);
    return transpose2;
  }

  // if the transposes have the same input and axis, they can folding into one,
  // the redundance should remove
  void FoldingTransposeHorizontal(
      const std::unordered_set<Instruction*>& all_transpose,
      const std::unordered_set<std::string>& fetch_ids,
      const InputToOpMap& in2instr,
      const OutputToOpMap& out2instr,
      std::unordered_set<Instruction*>* remove_instrs) const {
    std::unordered_map<TransposeKey, Variable*, TransposeKey::Hash>
        first_transpose_map;
    for (auto transpose : all_transpose) {
      if (("transpose" != (*transpose)->op_type) ||
          remove_instrs->count(transpose)) {
        continue;
      }

      const auto& input_id = (*transpose)->inputs.front()->id;
      const auto& output_id = (*transpose)->outputs.front()->id;
      const auto& axis = transpose->GetAttrs<ShapeType>("axis");

      TransposeKey key(input_id, axis);
      if (!first_transpose_map.count(key)) {
        VLOG(4) << "The transpose, whose output [" << output_id
                << "], cannot remove because it is the first transpose ! ";
        first_transpose_map.emplace(key, &(*transpose)->outputs.front());
        continue;
      }

      if (fetch_ids.find(output_id) != fetch_ids.end()) {
        // the transpose's output variable was fetched, skip
        VLOG(4) << "Cannot remove transpose, because the output [" << output_id
                << "] was fetched by other op ! ";
        continue;
      }

      VLOG(4) << "Try remove transpose, whose output [" << output_id << "]. ";
      remove_instrs->insert(transpose);

      const auto& output_ops = in2instr.at(output_id);
      for (auto op : output_ops) {
        ReplaceInputVariable(op, output_id, *first_transpose_map.at(key));
      }
    }
  }
};

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(TransposeCollapsing) {
  CINN_REGISTER_PROGRAM_PASS(TransposeCollapsing,
                             ::cinn::frontend::pass::TransposeCollapsingPass);

  return true;
}
