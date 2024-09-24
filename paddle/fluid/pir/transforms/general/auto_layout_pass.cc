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

#include "paddle/fluid/pir/transforms/general/auto_layout_pass.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/common/enforce.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/pir/dialect/operator/interface/layout_transformation.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pass/utils.h"

namespace {

class AutoLayoutPass : public pir::Pass {
 public:
  AutoLayoutPass() : pir::Pass("auto_layout_pass", 3) {}
  void Run(pir::Operation* op) override {
    for (size_t i = 0; i < op->num_regions(); ++i) {
      auto& region = op->region(i);
      for (auto& block : region) {
        pir::Builder builder = pir::Builder(ctx_, &block);
        VLOG(4) << "Transforming block";
        TransferLayout(builder, &block);
      }
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }

 private:
  void RewriteLayout(pir::Operation* op,
                     const std::vector<pir::Value>& input_values) {  // NOLINT
    auto InferMetaSpecificOp = [&]() {
      // Op not implement InferMetaInterface interface, so we need to rewrite
      // manually
      if (op->isa<pir::CombineOp>()) {
        auto out = op->dyn_cast<pir::CombineOp>().out();
        std::vector<pir::Type> new_out_type;
        for (auto v : op->operands_source()) {
          new_out_type.push_back(v.type());
        }
        auto new_out_type_v =
            pir::VectorType::get(pir::IrContext::Instance(), new_out_type);
        out.set_type(new_out_type_v);
      } else {
        PADDLE_THROW(common::errors::Unimplemented(
            "`%s` should implement InferMetaInterface interface or rewrite "
            "manually, but not found.",
            op->name()));
      }
    };

    if (op->HasAttribute("data_format")) {
      op->set_attribute("data_format", pir::StrAttribute::get(ctx_, "NHWC"));
    }
    auto p_attribute_map = op->attributes();

    if (auto infer_meta_interface =
            op->dyn_cast<paddle::dialect::InferMetaInterface>()) {
      auto output_types =
          infer_meta_interface.InferMeta(input_values, &p_attribute_map);
      for (size_t i = 0; i < output_types.size(); ++i) {
        op->result(i).set_type(output_types[i]);
        pir::SetNewLayoutForValue(op->result(i), common::DataLayout::NHWC);
      }
    } else {
      InferMetaSpecificOp();
    }
  }

  bool IsInsertTransposeOpBefore(pir::Operation* op) {
    bool is_insert_transpose = false;

    auto JudgeOperand = [&](const pir::Value& operand,
                            std::vector<int32_t> layout) {
      if (!JudgeValue(operand)) return false;
      auto transposeInputOp =
          operand.defining_op<paddle::dialect::TransposeOp>();
      if (!transposeInputOp) return false;
      const auto perm_attr =
          transposeInputOp.attribute<pir::ArrayAttribute>("perm");
      std::vector<int32_t> perm;
      for (size_t i = 0; i < perm_attr.size(); ++i) {
        auto attr = perm_attr.at(i);
        perm.push_back(attr.dyn_cast<pir::Int32Attribute>().data());
      }
      return perm == layout;
    };
    for (pir::Value operand : op->operands_source()) {
      if (operand.type().isa<pir::VectorType>()) {
        auto defined_op = operand.defining_op();
        for (auto inner_operand : defined_op->operands_source()) {
          is_insert_transpose = JudgeOperand(inner_operand, NHWC2NCHW_);
          if (is_insert_transpose) break;
        }
      } else {
        is_insert_transpose = JudgeOperand(operand, NHWC2NCHW_);
      }
      if (is_insert_transpose) break;
    }
    return is_insert_transpose;
  }

  void TransferLayout(pir::Builder builder, pir::Block* block) {
    for (auto&& op_item : *block) {
      auto op = &op_item;
      auto op_name = op->name();

      // Skip special ops.
      if (op->HasTrait<pir::ImmutableLayoutTrait>()) continue;
      if (op->operands().size() == 0) continue;

      // NHWC ops branch, Only support conv2d now, it will add white list later.
      if (op->isa<paddle::dialect::Conv2dOp>()) {
        if (op->HasAttribute("data_format") &&
            op->attribute<pir::StrAttribute>("data_format").AsString() ==
                "NCHW") {
          VLOG(4) << "enter NHWC op: " << op_name;
          DoTransposeOpOperand(op, builder);
          RewriteLayout(op, op->operands_source());
          DoTransposeOpResult(op, builder);
        }
      } else if (IsInsertTransposeOpBefore(op)) {
        VLOG(4) << "enter NCHW op: " << op_name;
        DoTransposeOpOperand(op, builder);
        RewriteLayout(op, op->operands_source());
        DoTransposeOpResult(op, builder);
      }
    }
  }

  // Skip the operand which is not dense tensor or not 4-D tensor, they don't
  // need transpose.
  bool JudgeValue(const pir::Value& value) {
    if (!value) {
      PADDLE_THROW(common::errors::Fatal(
          "value is null, please check the input tensor."));
    }
    if (!value.type()) {
      PADDLE_THROW(common::errors::Fatal(
          "value type is null, please check the input tensor type."));
    }
    if (auto type = value.type().dyn_cast<paddle::dialect::DenseTensorType>()) {
      return type.dims().size() == 4;
    }
    return false;
  }

  void DoTransposeOpOperand(pir::Operation* op,
                            pir::Builder& builder) {  // NOLINT
    builder.set_insertion_point(op);

    // For conv2d, only transpose the input.
    if (op->isa<paddle::dialect::Conv2dOp>()) {
      auto inp = op->operand(0);
      if (!JudgeValue(inp.source())) return;
      auto transpose_op =
          builder.Build<paddle::dialect::TransposeOp>(inp.source(), NCHW2NHWC_);
      pir::SetNewLayoutForValue(transpose_op->result(0),
                                common::DataLayout::NHWC);
      inp.set_source(transpose_op->result(0));
      return;
    }

    for (auto& operand : op->operands()) {
      if (!JudgeValue(operand.source())) continue;
      // Canbe optimize with cache when not eliminate the transpose op.
      auto transpose_op = builder.Build<paddle::dialect::TransposeOp>(
          operand.source(), NCHW2NHWC_);
      pir::SetNewLayoutForValue(transpose_op->result(0),
                                common::DataLayout::NHWC);
      operand.set_source(transpose_op->result(0));
    }
  }
  void DoTransposeOpResult(pir::Operation* op,
                           pir::Builder& builder) {  // NOLINT
    builder.SetInsertionPointAfter(op);
    for (auto& result : op->results()) {
      if (result.use_empty()) continue;
      if (!JudgeValue(result)) continue;
      auto transpose_op =
          builder.Build<paddle::dialect::TransposeOp>(result, NHWC2NCHW_);
      pir::SetNewLayoutForValue(transpose_op->result(0),
                                common::DataLayout::NCHW);
      result.ReplaceAllUsesWith(transpose_op->result(0));
      transpose_op->operand(0).set_source(result);
    }
  }
  pir::IrContext* ctx_ = pir::IrContext::Instance();
  const std::vector<int32_t> NCHW2NHWC_ = {0, 2, 3, 1};
  const std::vector<int32_t> NHWC2NCHW_ = {0, 3, 1, 2};
};
}  // namespace
namespace pir {

std::unique_ptr<Pass> CreateAutoLayoutPass() {
  return std::make_unique<AutoLayoutPass>();
}

}  // namespace pir

REGISTER_IR_PASS(auto_layout_pass, AutoLayoutPass);
