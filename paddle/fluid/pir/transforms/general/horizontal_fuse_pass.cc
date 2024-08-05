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
    for (size_t i = 0; i < op->num_results(); i++) {
      if (getOpCntUseX(op->result(i)) > 1) {
        return true;
      }
    }

    VLOG(4) << "horizontal_fuse_pass applied match on [" << op->name()
            << "] op";
    return false;
  }

  void Rewrite(pir::Operation* op,
               pir::PatternRewriter& rewriter) const override {  // NOLINT
    VLOG(4) << "horizontal_fuse_pass applies rewrite on [" << op->name()
            << "] op";

    for (size_t i = 0; i < op->num_results(); i++) {
      if (getOpCntUseX(op->result(i)) > 1) {
        rewriteOpsbyValue(op, rewriter, i);
      }
    }

    VLOG(4) << "horizontal_fuse_pass applied rewrite on [" << op->name()
            << "] op";
  }

 private:
  bool areAttributeMapsEqual(const pir::AttributeMap& attrs1,
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

  int getOpCntUseX(const pir::Value& x) const {
    /// 使用该出边的op中至少得有两个完全相同的MatmulOp/GemmEpilogueOp/FcOp
    /// 属性也应该完全相等，如果不完全相等，则不进行横向融合
    // 这就是一道算法题，我们直接构建屎山
    int op_cnt_useX = 0;

    // 我们假定，使用x的多个op的种类里，MatmulOp GemmEpilogueOp FcOp两两互斥
    // 即，最多有一种Op会出现。如果同时出现两种，则匹配失败。
    // 我们假定，该种类的多个op的属性会完全相同
    // 即不会出现某几个op属性相同，且和其他op的属性不一样的情况
    // 如果出现这个情况，直接匹配失败，不进行融合。
    pir::Operation* op_example = nullptr;
    std::string op_example_name;
    pir::AttributeMap op_example_attrs;
    for (auto it = x.use_begin(); it != x.use_end(); ++it) {
      pir::Operation* curr_op = it.owner();
      if (!(curr_op->isa<paddle::dialect::MatmulOp>() ||
            curr_op->isa<paddle::dialect::GemmEpilogueOp>() ||
            curr_op->isa<paddle::dialect::FcOp>())) {
        continue;
      }
      if (op_example == nullptr) {
        op_example = curr_op;
        op_example_name = curr_op->name();
        op_example_attrs = curr_op->attributes();
        op_cnt_useX++;
      } else {
        if (curr_op->name() == op_example_name &&
            areAttributeMapsEqual(op_example_attrs, curr_op->attributes())) {
          op_cnt_useX++;
        } else {
          return 0;
        }
      }
    }
    return op_cnt_useX;
  }

  void rewriteOpsbyValue(pir::Operation* op,
                         pir::PatternRewriter& rewriter,
                         size_t idx) const {
    /// 现在我们知道，x被多个op使用，该op属于MatmulOp/GemmEpilogueOp/FcOp的一种
    /// 我们统称 fused_matmul_op
    /// 数据准备
    // 准备x
    pir::Value x = op->result(idx);
    std::vector<int64_t> x_dims = pir::GetShapeFromValue(x);
    auto x_last_axis = x_dims.size() - 1;
    // 准备源模式算子
    std::vector<pir::Operation*> fused_matmul_ops;
    std::string fused_matmul_op_name;
    pir::AttributeMap fused_matmul_op_attrs;
    // 准备weight
    std::vector<int64_t> w_shapes;
    std::vector<pir::Value> combine_op_inputs_weight;
    // 准备bias
    bool with_bias = false;
    std::vector<pir::Value> combine_op_inputs_bias;
    // 准备outs
    std::vector<pir::Value> src_outs;

    for (auto it = x.use_begin(); it != x.use_end(); ++it) {
      pir::Operation* curr_op = it.owner();
      if (!(curr_op->isa<paddle::dialect::MatmulOp>() ||
            curr_op->isa<paddle::dialect::GemmEpilogueOp>() ||
            curr_op->isa<paddle::dialect::FcOp>())) {
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
        with_bias =
            (fused_matmul_op_name == paddle::dialect::GemmEpilogueOp::name() ||
             fused_matmul_op_name == paddle::dialect::FcOp::name());
      }
      if (with_bias) {
        combine_op_inputs_bias.push_back(curr_op->operand_source(2));
      }
    }

    /// 构建新图：（顺序插入新节点，逆序删除旧节点）
    /// 插入新节点
    // 我感觉如下的三句应该有简单写法，或许是不用挨个SetInsertionPointAfter
    // 但是我没找
    rewriter.SetInsertionPointAfter(op);
    auto combine_op_weight =
        rewriter.Build<pir::CombineOp>(combine_op_inputs_weight);
    auto weight_combined = combine_op_weight.out();  // pir::Value

    rewriter.SetInsertionPointAfter(combine_op_weight);
    auto concat_op_weight =
        rewriter.Build<paddle::dialect::ConcatOp>(weight_combined, 1);
    auto w_qkv = concat_op_weight.result(0);

    pir::Value xw_fused;
    pir::Operation* fused_matmul_op = nullptr;
    rewriter.SetInsertionPointAfter(concat_op_weight);
    if (!with_bias) {
      fused_matmul_op = rewriter.Build<paddle::dialect::MatmulOp>(x, w_qkv);
    } else {
      auto combine_bias =
          rewriter.Build<pir::CombineOp>(combine_op_inputs_bias);
      auto bias_combined = combine_bias.out();  // pir::Value

      rewriter.SetInsertionPointAfter(combine_bias);
      auto concat_op_bias =
          rewriter.Build<paddle::dialect::ConcatOp>(bias_combined, 0);
      auto bias_qkv = concat_op_bias.result(0);

      rewriter.SetInsertionPointAfter(concat_op_bias);
      if (fused_matmul_op_name == paddle::dialect::GemmEpilogueOp::name()) {
        fused_matmul_op = rewriter.Build<paddle::dialect::GemmEpilogueOp>(
            x, w_qkv, bias_qkv, fused_matmul_op_attrs);
      } else {
        fused_matmul_op = rewriter.Build<paddle::dialect::FcOp>(
            x, w_qkv, bias_qkv, fused_matmul_op_attrs);
      }
    }
    xw_fused = fused_matmul_op->result(0);

    rewriter.SetInsertionPointAfter(fused_matmul_op);
    auto split_op = rewriter.Build<paddle::dialect::SplitOp>(
        xw_fused, w_shapes, x_last_axis);
    auto xw_splitted = split_op.result(0);

    rewriter.SetInsertionPointAfter(split_op);
    auto split_builtin_op = rewriter.Build<pir::SplitOp>(xw_splitted);
    std::vector<pir::Value> xw = split_builtin_op.outputs();

    /// 替换结果Value
    for (size_t k = 0; k < src_outs.size(); k++) {
      rewriter.ReplaceAllUsesWith(src_outs[k], xw[k]);
    }
    /// 删除旧节点
    for (auto fused_matmul_op : fused_matmul_ops)
      rewriter.EraseOp(fused_matmul_op);
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

// 如果这里注册了，然后再跟USE_PIR_PASS(horizontal_fuse_pass);
// 是不是可以不修改analysis_predictor.cc
// 是否需要放到GPU路径下？或者跟死代码消除一样处理呢？
// REGISTER_IR_PASS(horizontal_fuse_pass, HorizontalFusePass);
