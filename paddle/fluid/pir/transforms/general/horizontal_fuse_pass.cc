// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/general/horizontal_fuse_pass.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/common/errors.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/parameter.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/region.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/pass/pass.h"
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

  bool Match(pir::Operation* op) const override {
    VLOG(4) << "horizontal_fuse_pass applies match on [" << op->name()
            << "] op";

    /// 至少有一个出边被多个op使用才能横向融合
    bool has_multiple_uses = false;
    /// 暂时假定只有一个出边被多个op使用。如果有多个，目前只处理最后一个（- -
    /// ！）
    int multiple_use_res_idx = -1;
    for (uint32_t i = 0; i < op->num_results(); i++) {
      if (op->result(i).use_count() > 1) {
        has_multiple_uses = true;
        multiple_use_res_idx = i;
      }
    }
    if (!has_multiple_uses) {
      return false;
    }
    // 这是一个很不规范的判断。意思是，默认matmul的输入x 是当前op的第一个输出。
    if (multiple_use_res_idx != 0) {
      return false;
    }
    pir::Value matmul_operand_x = op->result(multiple_use_res_idx);
    /// 如果使用各个出边的op不是matmul，则不能横向融合（暂时）
    for (auto it = matmul_operand_x.use_begin();
         it != matmul_operand_x.use_end();
         ++it) {
      if (!it.owner()->isa<paddle::dialect::MatmulOp>()) return false;
    }

    VLOG(4) << "horizontal_fuse_pass applied match on [" << op->name()
            << "] op";
    return true;
  }

  void Rewrite(pir::Operation* op,
               pir::PatternRewriter& rewriter) const override {  // NOLINT
    VLOG(4) << "horizontal_fuse_pass applies rewrite on [" << op->name()
            << "] op";

    /// 数据准备
    // 准备x
    pir::Value x = op->result(0);
    std::vector<int64_t> x_dims = pir::GetShapeFromValue(x);
    auto x_last_axis = x_dims.size() - 1;
    pir::Type x_dtype = pir::GetDataTypeFromValue(x);
    // GemmEpilogueOp只支持fp16和bf16
    bool is_fp16_or_bf16 =
        (x_dtype.isa<pir::BFloat16Type>() || x_dtype.isa<pir::Float16Type>());
    // 准备weight
    std::vector<int64_t> w_shapes;
    std::vector<pir::Value> combine_op_inputs_weight;
    std::vector<pir::Operation*> matmul_ops;
    // 准备bias
    int matmul_cnt = static_cast<int>(x.use_count());
    std::vector<pir::Value> combine_op_inputs_bias;
    std::vector<pir::Operation*> add_ops;
    bool with_bias = check_bias(x, matmul_cnt, combine_op_inputs_bias, add_ops);
    // 准备act
    bool with_act = false;
    std::vector<pir::Operation*> act_ops;
    if (with_bias) {
      with_act = check_act(x, matmul_cnt, act_ops, is_fp16_or_bf16);
    }
    // 准备outs
    std::vector<pir::Value> origin_outs;
    for (auto it_matmul = x.use_begin(); it_matmul != x.use_end();
         ++it_matmul) {
      auto curr_matmul_op = it_matmul.owner();
      matmul_ops.push_back(curr_matmul_op);
      combine_op_inputs_weight.push_back(curr_matmul_op->operand_source(1));
      auto w_dims = pir::GetShapeFromValue(curr_matmul_op->operand_source(1));
      w_shapes.push_back(w_dims[w_dims.size() - 1]);
      // if else嵌套 应该考虑怎么优化了。
      if (!with_bias) {
        origin_outs.push_back(curr_matmul_op->result(0));
      } else {
        auto curr_add_op = curr_matmul_op->result(0).first_use().owner();
        if (!with_act) {
          origin_outs.push_back(curr_add_op->result(0));
        } else {
          auto curr_act_op = curr_add_op->result(0).first_use().owner();
          origin_outs.push_back(curr_act_op->result(0));
        }
      }
    }

    /// 构建新图：（顺序插入新节点，逆序删除旧节点）
    // 插入新节点
    rewriter.SetInsertionPointAfter(op);
    auto combine_op_weight =
        rewriter.Build<pir::CombineOp>(combine_op_inputs_weight);
    auto weight_combined = combine_op_weight.out();  // pir::Value

    rewriter.SetInsertionPointAfter(combine_op_weight);
    auto concat_op_weight =
        rewriter.Build<paddle::dialect::ConcatOp>(weight_combined, 1);
    auto w_qkv = concat_op_weight.result(0);

    pir::Value x_qkv;
    pir::Operation* last_op = nullptr;
    rewriter.SetInsertionPointAfter(concat_op_weight);
    if (!with_bias) {
      auto matmul_op = rewriter.Build<paddle::dialect::MatmulOp>(x, w_qkv);
      x_qkv = matmul_op.result(0);
      last_op = matmul_op;
    } else {
      auto combine_bias =
          rewriter.Build<pir::CombineOp>(combine_op_inputs_bias);
      auto bias_combined = combine_bias.out();  // pir::Value

      rewriter.SetInsertionPointAfter(combine_bias);
      auto concat_op_bias =
          rewriter.Build<paddle::dialect::ConcatOp>(bias_combined, 0);
      auto bias_qkv = concat_op_bias.result(0);

      rewriter.SetInsertionPointAfter(concat_op_bias);
      pir::Operation* gemm_epilogue_op = nullptr;
      if (is_fp16_or_bf16) {
        gemm_epilogue_op = rewriter.Build<paddle::dialect::GemmEpilogueOp>(
            x, w_qkv, bias_qkv, x_last_axis);  // , "silu"
      } else {
        gemm_epilogue_op = rewriter.Build<paddle::dialect::FcOp>(
            x, w_qkv, bias_qkv, x_last_axis);
      }
      // 直接干懵了,  pir::Operation *?  pir::Operation?
      x_qkv = gemm_epilogue_op->result(0);

      // 这里直接可以融进GemmEpilogueOp里。
      if (!with_act) {
        last_op = gemm_epilogue_op;
      } else {
        pir::Operation* act_op = nullptr;
        if (act_ops[0]->isa<paddle::dialect::GeluOp>())
          act_op = rewriter.Build<paddle::dialect::GeluOp>(x_qkv);
        else if (act_ops[0]->isa<paddle::dialect::ReluOp>())
          act_op = rewriter.Build<paddle::dialect::ReluOp>(x_qkv);
        else if (act_ops[0]->isa<paddle::dialect::SiluOp>())
          act_op = rewriter.Build<paddle::dialect::SiluOp>(x_qkv);
        else
          throw std::runtime_error("Unsupported activation type");
        x_qkv = act_op->result(0);
        last_op = act_op;
      }
    }

    // 这里应该有last_op判空?
    rewriter.SetInsertionPointAfter(last_op);
    auto split_op =
        rewriter.Build<paddle::dialect::SplitOp>(x_qkv, w_shapes, x_last_axis);
    auto x_qkv_splitted = split_op.result(0);

    rewriter.SetInsertionPointAfter(split_op);
    auto split_builtin_op = rewriter.Build<pir::SplitOp>(x_qkv_splitted);
    std::vector<pir::Value> xq_xk_xv = split_builtin_op.outputs();

    assert(xq_xk_xv.size() == origin_outs.size());
    for (size_t k = 0; k < origin_outs.size(); k++) {
      rewriter.ReplaceAllUsesWith(origin_outs[k], xq_xk_xv[k]);
    }
    // 删除旧节点
    if (with_act) {
      for (auto act_op : act_ops) rewriter.EraseOp(act_op);
    }
    if (with_bias) {
      for (auto add_op : add_ops) rewriter.EraseOp(add_op);
    }
    for (auto matmul_op : matmul_ops) rewriter.EraseOp(matmul_op);

    VLOG(4) << "horizontal_fuse_pass applied rewrite on [" << op->name()
            << "] op";
  }

 private:
  bool check_bias(const pir::Value x,
                  int matmul_cnt,
                  std::vector<pir::Value>& combine_op_inputs_bias,
                  std::vector<pir::Operation*>& add_ops) const {
    for (auto it_matmul = x.use_begin(); it_matmul != x.use_end();
         ++it_matmul) {
      auto curr_matmul_op = it_matmul.owner();
      if (!check_bias_single(curr_matmul_op)) return false;

      auto curr_add_op = curr_matmul_op->result(0).first_use().owner();
      add_ops.push_back(curr_add_op);
      combine_op_inputs_bias.push_back(curr_add_op->operand_source(1));
    }
    return true;
  }

  /// 目前只支持处理 一个matmul + 一个bias 的情况
  bool check_bias_single(pir::Operation* op) const {
    if (op->num_results() != 1) {
      return false;
    }
    auto value_matmul_out = op->result(0);
    auto add_op = value_matmul_out.first_use().owner();
    if (!add_op->isa<paddle::dialect::AddOp>()) {
      return false;
    }
    if (pir::GetShapeFromValue(add_op->operand_source(1)).size() != 1) {
      return false;
    }
    return true;
  }

  bool check_act(const pir::Value x,
                 int matmul_cnt,
                 std::vector<pir::Operation*>& act_ops,
                 bool is_fp16_or_bf16) const {
    /// 在已知一个matmul + 一个bias 的情况下
    for (auto it_matmul = x.use_begin(); it_matmul != x.use_end();
         ++it_matmul) {
      auto curr_matmul_op = it_matmul.owner();
      auto curr_add_op = curr_matmul_op->result(0).first_use().owner();
      if (!check_act_single(curr_add_op, is_fp16_or_bf16)) return false;

      auto curr_act_op = curr_add_op->result(0).first_use().owner();
      act_ops.push_back(curr_act_op);
    }
    return true;
  }
  /// 目前只支持处理 一个matmul + 一个bias + 一个silu/relu/gelu 的情况
  bool check_act_single(pir::Operation* op, bool is_fp16_or_bf16) const {
    if (op->num_results() != 1) {
      return false;
    }
    auto value_addBias_out = op->result(0);
    auto act_op = value_addBias_out.first_use().owner();
    /// 这个是GemmEpilogueOp确定要融silu，才开放的版本。
    // if(!(act_op->isa<paddle::dialect::ReluOp>() ||
    //      (act_op->isa<paddle::dialect::GeluOp>() && is_fp16_or_bf16)||
    //      (act_op->isa<paddle::dialect::SiluOp>() && is_fp16_or_bf16))){
    //   return false;
    // }
    if (!(act_op->isa<paddle::dialect::ReluOp>() ||
          act_op->isa<paddle::dialect::GeluOp>() ||
          act_op->isa<paddle::dialect::SiluOp>())) {
      return false;
    }
    return true;
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
