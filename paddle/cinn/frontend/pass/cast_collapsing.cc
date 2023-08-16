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

class CastKey {
 public:
  CastKey(const std::string& input_id, const std::string& cast_type) {
    SetKey(input_id, cast_type);
  }

  void SetKey(const std::string& input_id, const std::string& cast_type) {
    input_id_ = input_id;
    cast_type_ = cast_type;
  }

  bool operator==(const CastKey& other) const {
    return cast_type_ == other.cast_type_ && input_id_ == other.input_id_;
  }
  bool operator!=(const CastKey& other) const {
    return !this->operator==(other);
  }

  struct Hash {
    size_t operator()(const CastKey& key) const {
      return std::hash<std::string>()(key.input_id_ + key.cast_type_);
    }
  };

 private:
  std::string input_id_;
  std::string cast_type_;
};

// Pass `CastCollapsing` folds multi cast into one.
class CastCollapsingPass : public ProgramPass {
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
    // all cast op in program
    std::unordered_set<Instruction*> all_cast;

    for (size_t i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];
      for (const auto& out : instr->outputs) {
        out2instr[out->id] = &instr;
      }
      for (const auto& in : instr->inputs) {
        in2instr[in->id].insert(&instr);
      }
      if ("cast" == instr->op_type) {
        all_cast.insert(&instr);
      }
    }

    // the useless cast op need to remove from program
    std::unordered_set<Instruction*> remove_instrs;
    FoldingCastVertical(
        all_cast, fetch_ids, in2instr, out2instr, &remove_instrs);

    for (auto instr : remove_instrs) {
      if (all_cast.count(instr)) {
        all_cast.erase(instr);
      }
    }
    // TODO(thisjiang): reopen after CINN support recompute for performance
    // due to recompute unsupported, if the op output to two group, it will also
    // create a new group, so that the horizontal fuse will not improve
    // performance. FoldingCastHorizontal(all_cast, fetch_ids, in2instr,
    // out2instr, &remove_instrs);

    NetBuilder builder("cast_collapsing_builder");
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
  void FoldingCastVertical(
      const std::unordered_set<Instruction*>& all_cast,
      const std::unordered_set<std::string>& fetch_ids,
      const InputToOpMap& in2instr,
      const OutputToOpMap& out2instr,
      std::unordered_set<Instruction*>* remove_instrs) const {
    if (all_cast.size() == 1) {
      return;
    }
    // the cast op should not remove
    std::unordered_set<Instruction*> visited_instrs;
    for (auto cast : all_cast) {
      if (("cast" != (*cast)->op_type) || visited_instrs.count(cast)) {
        // the cast op had been fused, skip
        continue;
      }

      // Fuse cast from front to back, the fuse path is unique
      auto first_cast = FindFirstCast(cast, out2instr);
      TryFuseCast(
          first_cast, fetch_ids, in2instr, remove_instrs, &visited_instrs);
    }
  }

  Instruction* FindFirstCast(Instruction* cast,
                             const OutputToOpMap& out2instr) const {
    auto first_cast = cast;

    auto input_name = (*first_cast)->inputs.front()->id;
    // Q: Why check whether cast's input in out2instr ?
    // A: The input may be the input of the graph other than another op's
    // output.
    //    Obviously, the cast op is the first cast in the situation.
    while (out2instr.count(input_name)) {
      auto instr = out2instr.at(input_name);
      if ("cast" != (*instr)->op_type) {
        // if input of cast is not output of another cast, it is the first cast.
        break;
      }

      input_name = (*instr)->inputs.front()->id;
      first_cast = instr;
    }
    return first_cast;
  }

  void TryFuseCast(Instruction* cast,
                   const std::unordered_set<std::string>& fetch_ids,
                   const InputToOpMap& in2instr,
                   std::unordered_set<Instruction*>* remove_instrs,
                   std::unordered_set<Instruction*>* visited_instrs) const {
    visited_instrs->insert(cast);

    const auto& input = (*cast)->inputs.front();
    const auto& input_name = input->id;
    const auto& input_dtype = input->type;

    const auto& output = (*cast)->outputs.front();
    const auto& output_name = output->id;
    const auto& output_dtype = output->type;

    const auto& dtype = cast->GetAttrs<std::string>("dtype");

    const auto& cast_info =
        output_name + "=cast(" + input_name + ", dtype=" + dtype + ")";

    bool can_remove = !fetch_ids.count(output_name);

    if (CheckCastBorder(cast, in2instr)) {
      if (can_remove) {
        VLOG(4) << "The op " << cast_info
                << " is a output op of graph, cannot fuse, remove.";
        // this cast not used by any other op, remove
        remove_instrs->insert(cast);
      } else {
        if (input_dtype == output_dtype) {
          VLOG(4) << "The cast op " << cast_info
                  << " is fetched but useless, replace with identity.";
          // cannot remove, however, the transpose is useless, we can replace
          // the cast with identity for more fusion opportunity
          ReplaceWithIdentity(cast);
        }
        // else the transpose is fetched and helpful, ignore
      }
      return;
    }

    // CheckCastBorder ensure `output_name` existed in `in2instr`
    const auto& out_instrs = in2instr.at(output_name);
    if (input_dtype == output_dtype) {
      if (!can_remove) {
        VLOG(4) << "The cast op " << cast_info
                << " is useless but fetched, replace with identity.";
        // cannot remove, but we can replace the cast with indentiy for more
        // fusion opportunity
        ReplaceWithIdentity(cast);
      } else {
        VLOG(4) << "The cast op " << cast_info << " is useless, remove.";
        for (auto instr : out_instrs) {
          // replace the input to cast's input
          ReplaceInputVariable(instr, output_name, input);
        }
        remove_instrs->insert(cast);

        for (auto instr : out_instrs) {
          if ("cast" == (*instr)->op_type) {
            // if the next instruction is cast op, continue fuse
            TryFuseCast(
                instr, fetch_ids, in2instr, remove_instrs, visited_instrs);
          }
        }
      }
      return;
    }

    if (!CheckOutputContainCast(cast, in2instr)) {
      VLOG(4) << "The cast op " << cast_info
              << " doesn't has output link to cast, skip.";
      return;
    }

    std::unordered_set<Instruction*> next_fused_instrs;

    for (auto instr : out_instrs) {
      if ("cast" != (*instr)->op_type) {
        // the cast was used by other non-cast op, cannot remove, skip
        can_remove = false;
        VLOG(4) << "Fuse cast of " << cast_info << " was used by "
                << (*instr)->op_type << ", cannot remove.";
        continue;
      }

      const auto& next_dtype = instr->GetAttrs<std::string>("dtype");

      VLOG(4) << "Fuse cast of " << cast_info << " and cast of "
              << (*instr)->outputs.front()->id << "=cast("
              << (*instr)->inputs.front()->id << ", dtype=" << next_dtype << ")"
              << " into cast of " << (*instr)->outputs.front()->id << "=cast("
              << input_name << ", dtype=" << next_dtype << ")";

      auto fused_cast = FuseCastImpl(cast, instr, next_dtype);

      next_fused_instrs.insert(fused_cast);
    }

    if (can_remove) {
      VLOG(4) << "Remove cast of " << cast_info;
      remove_instrs->insert(cast);
    }

    for (auto instr : next_fused_instrs) {
      TryFuseCast(instr, fetch_ids, in2instr, remove_instrs, visited_instrs);
    }
  }

  // check whether the op is the border op of graph, in other words, its output
  // var was not used by any op in graph.
  bool CheckCastBorder(Instruction* cast, const InputToOpMap& in2instr) const {
    const auto& output_name = (*cast)->outputs.front()->id;
    return !in2instr.count(output_name) || in2instr.at(output_name).empty();
  }

  // check whether the op's output ops has cast, if not, no cast need folding
  bool CheckOutputContainCast(Instruction* cast,
                              const InputToOpMap& in2instr) const {
    const auto& output_name = (*cast)->outputs.front()->id;
    for (auto instr : in2instr.at(output_name)) {
      if ("cast" == (*instr)->op_type) {
        return true;
      }
    }
    // the first cast's output is not anyone cast's input
    return false;
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

  // fuse the two cast dtype into the second cast, replace its input and dtype
  Instruction* FuseCastImpl(Instruction* cast1,
                            Instruction* cast2,
                            const std::string& fused_dtype) const {
    (*cast2)->inputs.front() = (*cast1)->inputs.front();
    cast2->SetAttr("dtype", fused_dtype);
    return cast2;
  }

  // if the casts have the same input and dtype, they can folding into one, the
  // redundance should remove
  void FoldingCastHorizontal(
      const std::unordered_set<Instruction*>& all_cast,
      const std::unordered_set<std::string>& fetch_ids,
      const InputToOpMap& in2instr,
      const OutputToOpMap& out2instr,
      std::unordered_set<Instruction*>* remove_instrs) const {
    std::unordered_map<CastKey, Variable*, CastKey::Hash> first_cast_map;
    for (auto cast : all_cast) {
      if (("cast" != (*cast)->op_type) || remove_instrs->count(cast)) {
        continue;
      }

      const auto& input_id = (*cast)->inputs.front()->id;
      const auto& output_id = (*cast)->outputs.front()->id;
      const auto& dtype = cast->GetAttrs<std::string>("dtype");

      CastKey key(input_id, dtype);
      if (!first_cast_map.count(key)) {
        VLOG(4) << "The cast, whose output [" << output_id
                << "], cannot remove because it is the first cast ! ";
        first_cast_map.emplace(key, &(*cast)->outputs.front());
        continue;
      }

      if (fetch_ids.find(output_id) != fetch_ids.end()) {
        // the cast's output variable was fetched, skip
        VLOG(4) << "Cannot remove cast, because the output [" << output_id
                << "] was fetched by other op ! ";
        continue;
      }

      VLOG(4) << "Try remove cast, whose output [" << output_id << "]. ";
      remove_instrs->insert(cast);

      const auto& output_ops = in2instr.at(output_id);
      for (auto op : output_ops) {
        ReplaceInputVariable(op, output_id, *first_cast_map.at(key));
      }
    }
  }
};

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(CastCollapsing) {
  CINN_REGISTER_PROGRAM_PASS(CastCollapsing,
                             ::cinn::frontend::pass::CastCollapsingPass);

  return true;
}
