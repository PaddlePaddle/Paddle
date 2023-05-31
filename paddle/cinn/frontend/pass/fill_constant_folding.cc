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

#include <sstream>
#include <string>
#include <unordered_set>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/program_pass.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/cinn/utils/type_defs.h"

namespace cinn::frontend::pass {

using cinn::utils::Attribute;
using cinn::utils::DimType;
using cinn::utils::ShapeType;

class FillConstantKey {
 public:
  FillConstantKey(const ShapeType& shape,
                  Attribute value,
                  const std::string& dtype,
                  bool force_cpu) {
    SetKey(shape, value, dtype, force_cpu);
  }

  void SetKey(const ShapeType& shape,
              Attribute value,
              const std::string& dtype,
              bool force_cpu) {
    shape_ = shape;
    value_ = value;
    force_cpu_ = force_cpu;
    dtype_ = dtype;
  }

  bool operator==(const FillConstantKey& other) const {
    return shape_ == other.shape_ && value_ == other.value_ &&
           force_cpu_ == other.force_cpu_ && dtype_ == other.dtype_;
  }
  bool operator!=(const FillConstantKey& other) const {
    return !this->operator==(other);
  }

  struct Hash {
    size_t operator()(const FillConstantKey& key) const {
      std::ostringstream hash_str;

      std::for_each(key.shape_.begin(),
                    key.shape_.end(),
                    [&](const DimType& dim) { hash_str << dim; });

      hash_str << utils::Attribute2String(key.value_);
      hash_str << key.force_cpu_;
      hash_str << key.dtype_;

      return std::hash<std::string>()(hash_str.str());
    }
  };

 private:
  ShapeType shape_;
  Attribute value_;
  bool force_cpu_;
  std::string dtype_;
};

// Pass `FillConstantFolding` folds several same fill_constant into one.
// If output of fill_constant in `fetch_ids`, keep the operator.
class FillConstantFoldingPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;
  using InputToOpMap =
      std::unordered_map<std::string, std::unordered_set<Instruction*>>;

 protected:
  void Clear() override {}

  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) const override {
    auto in2instr = GetInputToOpMap(program);

    // `fill_constant_map` is used to represent the first fill_constant and its
    // output variable
    std::unordered_map<FillConstantKey, Variable*, FillConstantKey::Hash>
        fill_constant_map;
    // `remove_instrs` is used to represent Instructions of which type is
    // fill_constant to be deleted.
    std::unordered_set<Instruction*> remove_instrs;

    for (int i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];

      if ("fill_constant" != instr->op_type) {
        // not fill_constant op, skip
        continue;
      }

      CHECK_EQ(instr->outputs.size(), 1UL)
          << "The fill_constant op should has one, and only one output ! "
             "Please check.";

      const auto& shape = instr.GetAttrs<ShapeType>("shape");
      auto value = instr->attrs.at("value");
      const auto& dtype = instr.GetAttrs<std::string>("dtype");
      auto force_cpu = instr.GetAttrs<bool>("force_cpu");

      FillConstantKey key(shape, value, dtype, force_cpu);
      if (!fill_constant_map.count(key)) {
        VLOG(4) << "The fill_constant, whose output is Var ["
                << instr->outputs[0]->id
                << "], cannot remove because it is the first fill_costant ! ";
        // retain the first fill constant op node
        fill_constant_map.emplace(key, &instr->outputs[0]);
        continue;
      }

      if (fetch_ids.find(instr->outputs[0]->id) != fetch_ids.end()) {
        // the fill constant's output variable was fetched, skip
        VLOG(4) << "Cannot remove fill_constant, because Var ["
                << instr->outputs[0]->id << "] was fetched by other op ! ";
        continue;
      }

      VLOG(4) << "Try remove fill_constant, whose output is Var ["
              << instr->outputs[0]->id << "]. ";
      remove_instrs.insert(&instr);

      auto constant_name = instr->outputs[0]->id;
      ReLinkFillConstant(in2instr, constant_name, fill_constant_map.at(key));
    }

    NetBuilder builder("fill_constant_folding_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }
    for (int i = 0; i < program->size(); i++) {
      if (remove_instrs.end() != remove_instrs.find(&(*program)[i])) continue;
      builder.AppendInstruction((*program)[i]);
    }
    *program = builder.Build();
  }

 private:
  static InputToOpMap GetInputToOpMap(Program* program) {
    // `in2instr` is used to represent the mapping of Input to Instruction.
    InputToOpMap in2instr;

    for (int i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];

      for (const auto& in : instr->inputs) {
        in2instr[in->id].insert(&instr);
      }
    }
    return in2instr;
  }

  static void ReLinkFillConstant(const InputToOpMap& in2instr,
                                 const std::string& input_var_name,
                                 Variable* out_var) {
    if (!in2instr.count(input_var_name)) {
      LOG(WARNING) << "Var [" << input_var_name << "] not used by other op ! ";
      return;
    }

    VLOG(4) << "Try replace the input Var [" << input_var_name << "] to ["
            << (*out_var)->id
            << "], because the fill_constant will be folding.";

    const auto& output_ops = in2instr.at(input_var_name);
    for (auto op : output_ops) {
      auto find_input = [&](const std::string& input_name) {
        return std::find_if(
            (*op)->inputs.begin(),
            (*op)->inputs.end(),
            [&](const Variable& var) { return var->id == input_name; });
      };

      // Why Loop : To avoid the op's inputs are the same variable !
      for (auto it = find_input(input_var_name); it != (*op)->inputs.end();
           it = find_input(input_var_name)) {
        // erase previous fill_constant output var and replace to new
        // fill_constant output var
        auto next_it = (*op)->inputs.erase(it);
        // keep the input place same, it's very important
        (*op)->inputs.insert(next_it, *out_var);
      }
    }
  }
};

}  // namespace cinn::frontend::pass

CINN_REGISTER_HELPER(FillConstantFolding) {
  CINN_REGISTER_PROGRAM_PASS(FillConstantFolding,
                             ::cinn::frontend::pass::FillConstantFoldingPass);

  return true;
}
