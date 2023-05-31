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

#include <ios>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "glog/logging.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/program_pass.h"

namespace cinn {
namespace frontend {
namespace pass {

class GemmRewriterPass : public ProgramPass {
 public:
  using ProgramPass::ProgramPass;

 protected:
  void Clear() override {
    removed_instrs_.clear();
    origin2new_.clear();
    output2instr_.clear();
    var_used_count_.clear();
  }

  void ApplyImpl(Program* prog,
                 const std::unordered_set<std::string>& fetch_ids,
                 const common::Target& target) override {
    if (target.arch != Target::Arch::NVGPU || !prog->size()) {
      return;
    }

    CollectInfo(*prog);

    NetBuilder builder("gemm_rewriter_builder");
    for (auto& var : prog->GetInputs()) {
      builder.CreateInput(var);
    }
    for (int i = prog->size() - 1; i >= 0; i--) {
      auto& instr = prog->operator[](i);
      /*if (instr->op_type == "elementwise_add") {
        auto fused = DoGemmFusion(&builder, instr, fetch_ids);
        if (fused) {
          // the elementwise_add is fused in gemm, just skip it
          continue;
        }
      }*/
      if (!removed_instrs_.count(instr.get())) {
        builder.AppendInstruction(instr);
      }
    }
    *prog = builder.Build(true);

    // Use the cublas call instead of the single matmul
    RewriteSingleMatmul(prog);

    // relink old outputs to new outputs
    for (size_t i = 0; i < prog->size(); i++) {
      auto& inputs = (*prog)[i]->inputs;
      for (size_t j = 0; j < inputs.size(); j++) {
        if (origin2new_.count(inputs[j].get())) {
          inputs[j] = origin2new_.at(inputs[j].get());
        }
      }
    }
    ClearResources();
  }

 private:
  void CollectInfo(const Program& prog) {
    for (size_t i = 0; i < prog.size(); i++) {
      auto& instr = prog[i];
      for (auto& var : instr->outputs) {
        output2instr_.emplace(var.get(), instr);
      }
      for (auto& var : instr->inputs) {
        var_used_count_[var.get()]++;
      }
    }
  }

  // Fuse the pattern of `matmul + add`
  bool DoGemmFusion(NetBuilder* builder,
                    const Instruction& instr,
                    const std::unordered_set<std::string>& fetch_ids) {
    CHECK_EQ(instr->inputs.size(), 2)
        << "elementwise should have only two inputs";
    std::vector<Variable> inputs;
    bool trans_a = false;
    bool trans_b = false;
    bool trans_out = false;
    float alpha = 1.f;
    std::unordered_set<std::string> dot_instrs{"matmul", "cublas_matmul"};
    for (auto& var : instr->inputs) {
      auto it = output2instr_.find(var.get());
      if (it != output2instr_.end() && dot_instrs.count(it->second->op_type)) {
        // If the output var of matmul is consumed by more than one instruction
        // or a fetch var, just skip to fuse it.
        CHECK_GT(var_used_count_.count(var.get()), 0)
            << "The input(" << var->id << ")"
            << "should be included in var_used_count_. Please check the "
               "CollectInfo method.";
        if ((var_used_count_.at(var.get()) > 1) || fetch_ids.count(var->id)) {
          continue;
        }

        auto& matmul_instr = it->second;
        // check inputs of cublas_gemm
        auto& bias = instr->inputs[0].get() == var.get() ? instr->inputs[1]
                                                         : instr->inputs[0];
        auto& matmul_inputs = matmul_instr->inputs;
        int lhs_dim_size = matmul_inputs[0]->shape.size();
        int rhs_dim_size = matmul_inputs[1]->shape.size();
        int bias_dim_size = bias->shape.size();
        // only support the condition below:
        // 1) tow-dim matrix multiply, such as m * k, k * n
        // 2) three-dim tensor multiply, such as b * m * k, b * k * n
        if (!((lhs_dim_size == 2 || lhs_dim_size == 3) &&
              lhs_dim_size == rhs_dim_size && rhs_dim_size == bias_dim_size)) {
          continue;
        }
        // set inputs of cublas_gemm
        inputs = matmul_inputs;
        inputs.emplace_back(bias);
        // set attrs of cublas_gemm
        auto& attrs = matmul_instr->attrs;
        if (attrs.count("trans_a")) {
          trans_a = absl::get<bool>(attrs.at("trans_a"));
        }
        if (attrs.count("trans_b")) {
          trans_b = absl::get<bool>(attrs.at("trans_b"));
        }
        if (attrs.count("trans_out")) {
          trans_out = absl::get<bool>(attrs.at("trans_out"));
        }
        if (attrs.count("alpha")) {
          alpha = absl::get<float>(attrs.at("alpha"));
        }

        // After the fusion, matmul and elementwise_add should be removed.
        removed_instrs_.emplace(matmul_instr.get());
        removed_instrs_.emplace(instr.get());
        break;
      }
    }

    if (inputs.size() == 3) {
      VLOG(4) << "-- The trans_a of GEMM: " << std::boolalpha << trans_a;
      VLOG(4) << "-- The trans_b of GEMM: " << std::boolalpha << trans_b;
      VLOG(4) << "-- The trans_out of GEMM: " << std::boolalpha << trans_out;
      const auto& new_outs = builder->CustomInstr("cublas_gemm",
                                                  inputs,
                                                  {{"trans_a", trans_a},
                                                   {"trans_b", trans_b},
                                                   {"trans_out", trans_out},
                                                   {"alpha", alpha}});
      auto new_out = new_outs[0];
      auto old_out = instr.GetOutput(0);
      new_out.set_id(old_out->id);
      origin2new_.emplace(old_out.get(), new_out);
      return true;
    }

    CHECK_EQ(inputs.size(), 0) << "The gemm should only have three inputs.";
    return false;
  }

  // Rewrite the left single matmul, use cublas call instead
  void RewriteSingleMatmul(Program* prog) {
    for (int i = 0; i < prog->size(); i++) {
      auto& instr = (*prog)[i];
      if (instr->op_type == "matmul") {
        auto& matmul_inputs = instr->inputs;
        int lhs_dim_size = matmul_inputs[0]->shape.size();
        int rhs_dim_size = matmul_inputs[1]->shape.size();
        // only support the condition below:
        // 1) tow-dim matrix multiply, such as m * k, k * n
        // 2) three-dim tensor multiply, such as b * m * k, b * k * n
        if (lhs_dim_size <= 4 && rhs_dim_size <= 4) {
          instr->op_type = "cublas_matmul";
        }
      }
    }
  }

  void ClearResources() {
    removed_instrs_.clear();
    origin2new_.clear();
    output2instr_.clear();
    var_used_count_.clear();
  }

 private:
  std::unordered_set<_Instruction_*> removed_instrs_;
  std::unordered_map<_Variable_*, Variable> origin2new_;
  std::unordered_map<_Variable_*, Instruction> output2instr_;
  std::unordered_map<_Variable_*, int> var_used_count_;
};

}  // namespace pass
}  // namespace frontend
}  // namespace cinn

namespace fp = ::cinn::frontend::pass;
CINN_REGISTER_HELPER(GemmRewriter) {
  CINN_REGISTER_PROGRAM_PASS(GemmRewriter, fp::GemmRewriterPass);

  return true;
}
