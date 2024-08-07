// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/gpu/horizontal_fuse_pass.h"

#include <string>
#include <vector>

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/region.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

class HorizontalFusePattern : public pir::RewritePattern {
 public:
  explicit HorizontalFusePattern(pir::IrContext* context)
      : RewritePattern(MatchAnyOpTypeTag(),
                       1 /*benefit*/,
                       context,
                       {} /*generated_names*/) {}
  bool MatchAndRewrite(
      pir::Operation* op,
      pir::PatternRewriter& rewriter) const override {  // NOLINT
    bool match_flag = false;
    for (size_t i = 0; i < op->num_results(); i++) {
      if (GetOpCntUseX(op->result(i)) > 1) {
        match_flag = true;
        RewriteOpsbyValue(op, &rewriter, i);
        VLOG(4) << "horizontal_fuse_pass applied rewrite on [" << op->name()
                << "] op";
      }
    }
    return match_flag;
  }

 private:
  bool IsValidOp(pir::Operation* curr_op) const {
    if (curr_op->isa<paddle::dialect::GemmEpilogueOp>() ||
        curr_op->isa<paddle::dialect::FcOp>()) {
      return true;
    }

    if (curr_op->isa<paddle::dialect::MatmulOp>()) {
      if (curr_op->HasAttribute("transpose_x") &&
          curr_op->attribute<pir::BoolAttribute>("transpose_x").data() ==
              true) {
        return false;
      }
      if (curr_op->HasAttribute("transpose_y") &&
          curr_op->attribute<pir::BoolAttribute>("transpose_y").data() ==
              true) {
        return false;
      }
      return true;
    }
    return false;
  }

  bool AreAttributeMapsEqual(const pir::AttributeMap& attrs1,
                             const pir::AttributeMap& attrs2) const {
    if (attrs1.size() != attrs2.size()) {
      return false;
    }
    pir::AttributeMap attrs2_copy(attrs2);
    for (const auto& kv : attrs1) {
      auto it = attrs2_copy.find(kv.first);
      if (it == attrs2_copy.end() || it->second != kv.second) {
        return false;
      }
      attrs2_copy.erase(it);
    }
    return attrs2_copy.empty();
  }

  int GetOpCntUseX(const pir::Value& x) const {
    // At least two same MatmulOp/GemmEpilogueOp/FcOp using x
    // MatmulOp/GemmEpilogueOp/FcOp mutually exclusive and don't appear at the
    // same time. All ops'attrs should also be exactly equal. if not, fusion is
    // not performed.
    int op_cnt_use_x = 0;
    pir::Operation* op_example = nullptr;
    std::string op_example_name;
    uint32_t op_example_num_operands = 0;
    pir::AttributeMap op_example_attrs;
    for (auto it = x.use_begin(); it != x.use_end(); ++it) {
      pir::Operation* curr_op = it.owner();
      if (!IsValidOp(curr_op)) {
        continue;
      }
      if (op_example == nullptr) {
        op_example = curr_op;
        op_example_name = curr_op->name();
        op_example_num_operands = curr_op->num_operands();
        op_example_attrs = curr_op->attributes();
        op_cnt_use_x++;
      } else {
        if (curr_op->name() == op_example_name &&
            curr_op->num_operands() == op_example_num_operands &&
            AreAttributeMapsEqual(op_example_attrs, curr_op->attributes())) {
          op_cnt_use_x++;
        } else {
          return 0;
        }
      }
    }
    return op_cnt_use_x;
  }

  template <typename OpTy, typename... Args>
  pir::Value InsertOpAfter(pir::Operation** prev_op_ptr,
                           pir::PatternRewriter* rewriter,
                           Args&&... args) const {
    rewriter->SetInsertionPointAfter(*prev_op_ptr);
    pir::Operation* new_op = rewriter->Build<OpTy>(std::forward<Args>(args)...);
    *prev_op_ptr = new_op;
    return new_op->result(0);
  }

  void RewriteOpsbyValue(pir::Operation* op,
                         pir::PatternRewriter* rewriter,
                         size_t idx) const {
    /// x is used by the op, which belongs to a kind of
    /// MatmulOp/GemmEpilogueOp/FcOp
    // prepare x
    pir::Value x = op->result(idx);
    std::vector<int64_t> x_dims = pir::GetShapeFromValue(x);
    auto x_last_axis = x_dims.size() - 1;
    // prepare src ops
    std::vector<pir::Operation*> fused_matmul_ops;
    std::string fused_matmul_op_name;
    pir::AttributeMap fused_matmul_op_attrs;
    // prepare weight
    std::vector<int64_t> w_shapes;
    std::vector<pir::Value> combine_op_inputs_weight;
    // prepare bias
    std::vector<pir::Value> combine_op_inputs_bias;
    // prepare outs
    std::vector<pir::Value> src_outs;

    for (auto it = x.use_begin(); it != x.use_end(); ++it) {
      pir::Operation* curr_op = it.owner();
      if (!IsValidOp(curr_op)) {
        continue;
      }
      fused_matmul_ops.push_back(curr_op);
      combine_op_inputs_weight.push_back(curr_op->operand_source(1));
      auto w_dims = pir::GetShapeFromValue(curr_op->operand_source(1));
      w_shapes.push_back(w_dims[w_dims.size() - 1]);

      src_outs.push_back(curr_op->result(0));

      if (fused_matmul_op_name.empty()) {
        fused_matmul_op_name = curr_op->name();
        fused_matmul_op_attrs = curr_op->attributes();
      }
      if (curr_op->num_operands() > 2) {
        combine_op_inputs_bias.push_back(curr_op->operand_source(2));
      }
    }

    /// build new graph
    /// insert new ops
    pir::Operation* prev_op = op;
    auto weight_combined = InsertOpAfter<pir::CombineOp>(
        &prev_op, rewriter, combine_op_inputs_weight);
    auto w_qkv = InsertOpAfter<paddle::dialect::ConcatOp>(
        &prev_op, rewriter, weight_combined, 1);

    pir::Value bias_qkv;
    if (combine_op_inputs_bias.size()) {
      auto bias_combined = InsertOpAfter<pir::CombineOp>(
          &prev_op, rewriter, combine_op_inputs_bias);
      bias_qkv = InsertOpAfter<paddle::dialect::ConcatOp>(
          &prev_op, rewriter, bias_combined, 0);
    }

    pir::Value xw_fused;
    if (fused_matmul_op_name == paddle::dialect::MatmulOp::name()) {
      xw_fused = InsertOpAfter<paddle::dialect::MatmulOp>(
          &prev_op, rewriter, x, w_qkv, fused_matmul_op_attrs);
    } else if (fused_matmul_op_name ==
               paddle::dialect::GemmEpilogueOp::name()) {
      xw_fused = InsertOpAfter<paddle::dialect::GemmEpilogueOp>(
          &prev_op, rewriter, x, w_qkv, bias_qkv, fused_matmul_op_attrs);
    } else {
      xw_fused = InsertOpAfter<paddle::dialect::FcOp>(
          &prev_op, rewriter, x, w_qkv, bias_qkv, fused_matmul_op_attrs);
    }

    auto xw_splitted = InsertOpAfter<paddle::dialect::SplitOp>(
        &prev_op, rewriter, xw_fused, w_shapes, x_last_axis);
    rewriter->SetInsertionPointAfter(prev_op);
    auto split_builtin_op = rewriter->Build<pir::SplitOp>(xw_splitted);
    std::vector<pir::Value> xw = split_builtin_op.outputs();

    /// replace res Value
    for (size_t k = 0; k < src_outs.size(); k++) {
      rewriter->ReplaceAllUsesWith(src_outs[k], xw[k]);
    }
    /// del old ops
    for (auto fused_matmul_op : fused_matmul_ops)
      rewriter->EraseOp(fused_matmul_op);
  }
};

class HorizontalFusePass : public pir::Pass {
 public:
  HorizontalFusePass() : pir::Pass("horizontal_fuse_pass", 1) {}

  bool Initialize(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<HorizontalFusePattern>(context);
    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation* op) override {
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 10;
    auto [_, num_rewrites] = pir::ApplyPatternsGreedily(op, patterns_, cfg);
    AddStatistics(num_rewrites);
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateHorizontalFusePass() {
  return std::make_unique<HorizontalFusePass>();
}

}  // namespace pir

REGISTER_IR_PASS(horizontal_fuse_pass, HorizontalFusePass);
