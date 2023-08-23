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

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "glog/logging.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/program_pass.h"

namespace cinn {
namespace frontend {
namespace pass {

#define FILL_CONSTANT_VALUE_REWRITE(OLD_VALUE, FUNC, NEW_VALUE) \
  if (absl::holds_alternative<float>(OLD_VALUE))                \
    NEW_VALUE = FUNC(absl::get<float>(OLD_VALUE));              \
  else if (absl::holds_alternative<double>(OLD_VALUE))          \
    NEW_VALUE = FUNC(absl::get<double>(OLD_VALUE));             \
  else if (absl::holds_alternative<int>(OLD_VALUE))             \
    NEW_VALUE = FUNC(absl::get<int>(OLD_VALUE));                \
  else if (absl::holds_alternative<int64_t>(OLD_VALUE))         \
    NEW_VALUE = FUNC(absl::get<int64_t>(OLD_VALUE));            \
  else                                                          \
    LOG(FATAL) << "fill_constant Only support float32/float64/int32/int64";

#define MATH_FUNC_REWRITER(op_name)                                            \
  {                                                                            \
#op_name, [](const Instruction& fill_constant, Instruction* instr) -> void { \
       (*instr)->op_type = "fill_constant"; \
       (*instr)->inputs.clear(); \
       (*instr)->attrs = fill_constant->attrs; \
       const auto& old_attr = fill_constant->attrs.at("value"); \
       auto& new_attr = (*instr)->attrs.at("value"); \
       FILL_CONSTANT_VALUE_REWRITE(old_attr, std::op_name, new_attr) \
     } \
  }

static std::unordered_map<std::string,
                          std::function<void(const Instruction&, Instruction*)>>
    rewriter_ops = {
        {"reshape",
         [](const Instruction& fill_constant, Instruction* instr) -> void {
           (*instr)->op_type = "fill_constant";
           (*instr)->inputs.clear();
           // the outputs keep same

           CHECK((*instr)->attrs.count("shape"))
               << "The reshape op should has attribute [shape]!";
           auto new_shape = (*instr)->attrs.at("shape");
           (*instr)->attrs = fill_constant->attrs;
           (*instr)->attrs["shape"] = new_shape;
         }},
        {"scale",
         [](const Instruction& fill_constant, Instruction* instr) -> void {
           (*instr)->op_type = "fill_constant";
           (*instr)->inputs.clear();
           // the outputs keep same

           auto scale = (*instr)->attrs.count("scale")
                            ? instr->GetAttrs<float>("scale")
                            : 1.0f;
           auto bias = (*instr)->attrs.count("bias")
                           ? instr->GetAttrs<float>("bias")
                           : 0.0f;
           auto bias_after_scale =
               (*instr)->attrs.count("bias_after_scale")
                   ? instr->GetAttrs<bool>("bias_after_scale")
                   : true;

           (*instr)->attrs = fill_constant->attrs;

           const auto& old_attr = fill_constant->attrs.at("value");
           auto& new_attr = (*instr)->attrs.at("value");
           if (bias_after_scale) {
             auto scale_func = [&](const auto& value) -> decltype(auto) {
               return value * static_cast<decltype(value)>(scale) +
                      static_cast<decltype(value)>(bias);
             };
             FILL_CONSTANT_VALUE_REWRITE(old_attr, scale_func, new_attr)
           } else {
             auto scale_func = [&](const auto& value) -> decltype(auto) {
               return (value + static_cast<decltype(value)>(bias)) *
                      static_cast<decltype(value)>(scale);
             };
             FILL_CONSTANT_VALUE_REWRITE(old_attr, scale_func, new_attr)
           }
         }},
        {"cast",
         [](const Instruction& fill_constant, Instruction* instr) -> void {
           (*instr)->op_type = "fill_constant";
           (*instr)->inputs.clear();
           // the outputs keep same

           CHECK((*instr)->attrs.count("dtype"))
               << "The cast op should has attribute [dtype]!";
           auto cast_dtype = instr->GetAttrs<std::string>("dtype");

           (*instr)->attrs = fill_constant->attrs;
           (*instr)->attrs["dtype"] = cast_dtype;
         }},
        {"broadcast_to",
         [](const Instruction& fill_constant, Instruction* instr) -> void {
           (*instr)->op_type = "fill_constant";
           (*instr)->inputs.clear();
           // the outputs keep same

           CHECK((*instr)->attrs.count("out_shape"))
               << "The cast op should has attribute [out_shape]!";
           auto out_shape = instr->GetAttrs<std::vector<int>>("out_shape");

           (*instr)->attrs = fill_constant->attrs;
           (*instr)->attrs["shape"] = out_shape;
         }},
        {"slice",
         [](const Instruction& fill_constant, Instruction* instr) -> void {
           (*instr)->op_type = "fill_constant";
           (*instr)->inputs.clear();
           // the outputs keep same

           (*instr)->attrs = fill_constant->attrs;
           (*instr)->attrs["shape"] = (*instr)->outputs[0]->shape;
         }},
        MATH_FUNC_REWRITER(abs),
        MATH_FUNC_REWRITER(log),
        MATH_FUNC_REWRITER(log2),
        MATH_FUNC_REWRITER(log10),
        MATH_FUNC_REWRITER(tanh)};

#undef FILL_CONSTANT_VALUE_REWRITE
#undef MATH_FUNC_REWRITER

class FillConstantRewriterPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 protected:
  void Clear() override {}

  void ApplyImpl(Program* program,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    auto input2instr = GetInput2Instr(program);

    std::unordered_set<const Instruction*> remove_instr;
    for (int i = 0; i < program->size(); ++i) {
      const auto& instr = (*program)[i];

      if (instr->op_type == "fill_constant") {
        RewriteFillConstant(instr, input2instr, fetch_ids, &remove_instr);
      }
    }
    VLOG(3) << "FillConstantRewriterPass Remove " << remove_instr.size()
            << " instruction";

    NetBuilder builder("reshape_rewritter_builder");
    for (auto& var : program->GetInputs()) {
      builder.CreateInput(var);
    }

    for (int i = 0; i < program->size(); ++i) {
      const auto& instr = (*program)[i];

      if (!remove_instr.count(&instr)) {
        builder.AppendInstruction(instr);
      }
    }
    *program = builder.Build();
  }

 private:
  using Input2Instr =
      std::unordered_map<std::string, std::unordered_set<Instruction*>>;

  Input2Instr GetInput2Instr(Program* program) {
    Input2Instr input2instr;

    for (int i = 0; i < program->size(); ++i) {
      auto& instr = (*program)[i];
      for (const auto& var : instr->inputs) {
        input2instr[var->id].insert(&instr);
      }
    }

    return input2instr;
  }

  void RewriteFillConstant(
      const Instruction& fill_constant,
      const Input2Instr& input2instr,
      const std::unordered_set<std::string>& fetch_ids,
      std::unordered_set<const Instruction*>* remove_instr) {
    CHECK_EQ(fill_constant->op_type, std::string("fill_constant"));
    CHECK_EQ(fill_constant->outputs.size(), 1UL)
        << "The fill_constant op should just has one output! Please check.";
    const auto& out = fill_constant->outputs[0];

    if (!input2instr.count(out->id)) {
      // the fill constant's output is empty, skip
      return;
    }

    bool can_remove = true;
    for (auto* instr : input2instr.at(out->id)) {
      if (rewriter_ops.count((*instr)->op_type)) {
        VLOG(3) << "Try folding " << (*instr) << " into " << fill_constant;
        rewriter_ops.at((*instr)->op_type)(fill_constant, instr);
        RewriteFillConstant(*instr, input2instr, fetch_ids, remove_instr);
      } else {
        can_remove = false;
      }
    }

    if (can_remove && !fetch_ids.count(out->id)) {
      remove_instr->insert(&fill_constant);
    }
  }
};

}  // namespace pass
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(FillConstantRewriter) {
  CINN_REGISTER_PROGRAM_PASS(FillConstantRewriter,
                             cinn::frontend::pass::FillConstantRewriterPass);

  return true;
}
