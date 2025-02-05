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

#include "paddle/cinn/hlir/dialect/operator/transforms/accuracy_check_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_to_pd_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

COMMON_DECLARE_double(accuracy_check_rtol_fp32);  // default 1e-6
COMMON_DECLARE_double(accuracy_check_atol_fp32);  // default 1e-6
COMMON_DECLARE_double(accuracy_check_rtol_fp16);  // default 1e-3
COMMON_DECLARE_double(accuracy_check_atol_fp16);  // default 1e-3
COMMON_DECLARE_double(accuracy_check_rtol_bf16);  // default 1e-3
COMMON_DECLARE_double(accuracy_check_atol_bf16);  // default 1e-3

namespace cinn::dialect::ir {

static constexpr bool equal_nan = false;  // whether to consider NaN as equal

static void GetTolerance(pir::Value value, double* rtol, double* atol) {
  const auto& GetValueDtype = [&](pir::Value value) -> phi::DataType {
    if (value.type().isa<paddle::dialect::DenseTensorType>()) {
      return paddle::dialect::TransToPhiDataType(
          value.type().dyn_cast<paddle::dialect::DenseTensorType>().dtype());
    } else if (value.type().isa<paddle::dialect::SelectedRowsType>()) {
      return paddle::dialect::TransToPhiDataType(
          value.type().dyn_cast<paddle::dialect::SelectedRowsType>().dtype());
    } else if (value.type().isa<paddle::dialect::DenseTensorArrayType>()) {
      return paddle::dialect::TransToPhiDataType(
          value.type()
              .dyn_cast<paddle::dialect::DenseTensorArrayType>()
              .dtype());
    } else if (value.type().isa<paddle::dialect::SparseCooTensorType>()) {
      return paddle::dialect::TransToPhiDataType(
          value.type()
              .dyn_cast<paddle::dialect::SparseCooTensorType>()
              .dtype());
    } else if (value.type().isa<paddle::dialect::SparseCsrTensorType>()) {
      return paddle::dialect::TransToPhiDataType(
          value.type()
              .dyn_cast<paddle::dialect::SparseCsrTensorType>()
              .dtype());
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Currently, we can only get phi::DataType from DenseTensorType and "
          "SelectedRowsType."));
    }
  };

  if (GetValueDtype(value) == phi::DataType::FLOAT16) {
    *rtol = FLAGS_accuracy_check_rtol_fp16;
    *atol = FLAGS_accuracy_check_atol_fp16;
  } else if (GetValueDtype(value) == phi::DataType::BFLOAT16) {
    *rtol = FLAGS_accuracy_check_rtol_bf16;
    *atol = FLAGS_accuracy_check_atol_bf16;
  } else {
    *rtol = FLAGS_accuracy_check_rtol_fp32;
    *atol = FLAGS_accuracy_check_atol_fp32;
  }
}

class AddAccuracyCheckPattern
    : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  using pir::OpRewritePattern<cinn::dialect::FusionOp>::OpRewritePattern;

  bool MatchAndRewrite(cinn::dialect::FusionOp fusion_op,
                       pir::PatternRewriter& rewriter) const override {
    const auto op_list = fusion_op.GetOperators();

    const auto group_info = fusion_op.attribute("group_info")
                                .dyn_cast<cinn::dialect::GroupInfoAttribute>()
                                .data();
    const auto& fn_name = group_info.fn_name;

    ::pir::IrMapping ir_mapping;
    ::pir::CloneOptions clone_options(/*clone_regions=*/false,
                                      /*clone_operands=*/true,
                                      /*clone_successors=*/false);
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

    double rtol, atol;
    const auto& InsertAccuaryCheckOp = [&](::pir::Operation* op) -> void {
      for (size_t i = 0; i < op->num_operands(); ++i) {
        GetTolerance(fusion_op.result(i), &rtol, &atol);
        rewriter.Build<paddle::dialect::AccuracyCheckOp>(
            fusion_op.result(i),
            ir_mapping.Lookup(op->operand_source(i)),
            fn_name,
            rtol,
            atol,
            equal_nan);
      }
    };

    const auto& ConvertCinnOpToPdOp = [&](::pir::Operation* op) -> void {
      for (size_t i = 0; i < op->num_operands(); ++i) {
        if (!ir_mapping.GetMap<pir::Value>().count(op->operand_source(i))) {
          ir_mapping.Add(op->operand_source(i), op->operand_source(i));
        }
      }
      if (op->isa<cinn::dialect::GenerateShapeOp>()) {
        auto cloned_op = op->Clone(ir_mapping, clone_options);
        rewriter.Insert(cloned_op);
        auto cinn_op = cloned_op->dyn_cast<cinn::dialect::GenerateShapeOp>();
        std::optional<pir::Value> out_replacement =
            details::GetOutReplacement(cinn_op, &rewriter);
        if (!out_replacement.has_value()) return;

        rewriter.ReplaceAllUsesWith(cloned_op->result(0),
                                    out_replacement.value());
        PADDLE_ENFORCE_EQ(cloned_op->use_empty(),
                          true,
                          ::common::errors::InvalidArgument(
                              "cinn_op.generate_shape op shouldn't "
                              "be used outside fusion block."));
        rewriter.EraseOp(cloned_op);
        ir_mapping.Add(op->result(0), out_replacement.value());
        rewriter.SetInsertionPointAfter(out_replacement.value().defining_op());
        return;
      }
      if (op->isa<cinn::dialect::YieldStoreOp>()) {
        VLOG(6) << "skip yield_store op";
        ir_mapping.Add(op->result(0), ir_mapping.Lookup(op->operand_source(0)));
        return;
      }
      pir::Operation* pd_op =
          cinn::dialect::details::RewriteCinnOpToPdOp(op, ir_mapping, rewriter);
      rewriter.SetInsertionPointAfter(pd_op);
    };

    const auto& ClonePdOp = [&](::pir::Operation* op) -> void {
      for (size_t i = 0; i < op->num_operands(); ++i) {
        if (!ir_mapping.GetMap<pir::Value>().count(op->operand_source(i))) {
          ir_mapping.Add(op->operand_source(i), op->operand_source(i));
        }
      }
      auto new_op = op->Clone(ir_mapping, clone_options);
      rewriter.Insert(new_op);
      rewriter.SetInsertionPointAfter(new_op);
    };

    rewriter.SetInsertionPointAfter(fusion_op);
    for (auto& op : op_list) {
      if (op->isa<::pir::YieldOp>()) {
        InsertAccuaryCheckOp(op);
      } else if (op->dialect()->name() == "cinn_op") {
        ConvertCinnOpToPdOp(op);
      } else {
        ClonePdOp(op);
      }
    }
    return true;
  }
};

class AccuracyCheckPass : public pir::Pass {
 public:
  AccuracyCheckPass() : pir::Pass("accuracy_check_pass", /*opt_level=*/1) {}

  bool Initialize(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<AddAccuracyCheckPattern>(context);

    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation* op) override {
    int64_t num_ops{0};
    for (uint32_t i = 0; i < op->num_regions(); ++i) {
      auto& region = op->region(i);
      for (auto& block : region) {
        num_ops += block.size();
      }
    }
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 1;
    auto [_, num_rewrites] = pir::ApplyPatternsGreedily(op, patterns_, cfg);
    AddStatistics(num_rewrites, num_ops);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
};

std::unique_ptr<pir::Pass> CreateAccuracyCheckPass() {
  return std::make_unique<AccuracyCheckPass>();
}

}  // namespace cinn::dialect::ir
