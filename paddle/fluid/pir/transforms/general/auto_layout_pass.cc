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
  AutoLayoutPass() : pir::Pass("auto_layout_pass", 2) {}
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
    if (op->isa<paddle::dialect::ConcatOp>() ||
        op->isa<paddle::dialect::ArgmaxOp>()) {
      auto layout_interface =
          op->dyn_cast<paddle::dialect::LayoutTransformationInterface>();
      layout_interface.RewriteByLayout(op, common::DataLayout::NHWC);
      return;
    }

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

  bool JudgeOperand(const pir::Value& operand, std::vector<int32_t> layout) {
    if (operand.type().isa<pir::VectorType>()) {
      auto defined_op = operand.defining_op();
      for (auto inner_operand : defined_op->operands_source()) {
        return JudgeOperand(inner_operand, NCHW2NHWC_);
      }
    } else {
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
    }
  }

  bool IsInsertTransposeOpBefore(pir::Operation* op) {
    bool is_insert_transpose = false;

    for (pir::Value operand : op->operands_source()) {
      if (is_insert_transpose) break;
      is_insert_transpose = JudgeOperand(operand, NHWC2NCHW_);
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
      if (op->isa<paddle::dialect::Conv2dOp>() ||
          op->isa<paddle::dialect::FusedConv2dAddActOp>() ||
          op->isa<paddle::dialect::Conv2dTransposeOp>()) {
        if (op->HasAttribute("data_format") &&
            op->attribute<pir::StrAttribute>("data_format").AsString() ==
                "NCHW") {
          VLOG(4) << "enter NHWC op: " << op_name;
          DoTransposeOpOperand(op, builder);
          RewriteLayout(op, op->operands_source());
          DoTransposeOpResult(op, builder);
        }
      } else if (op_in_NCHW_.find(op_name) == op_in_NCHW_.end() &&
                 op_with_axis_.find(op_name) == op_with_axis_.end() &&
                 IsInsertTransposeOpBefore(op)) {
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
    if (!value) return false;
    if (!value.type()) return false;
    if (auto type = value.type().dyn_cast<paddle::dialect::DenseTensorType>()) {
      return type.dims().size() == 4;
    }
    return false;
  }

  void DoTransposeOpOperand(pir::Operation* op,
                            pir::Builder& builder) {  // NOLINT
    builder.set_insertion_point(op);

    // For conv2d, only transpose the input.
    if (op->isa<paddle::dialect::Conv2dOp>() ||
        op->isa<paddle::dialect::Conv2dTransposeOp>()) {
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
  const std::set<std::string> op_in_NCHW_ = {"pd_op.max_pool2d_with_index",
                                             "pd_op.fractional_max_pool2d",
                                             "pd_op.unpool3d",
                                             "pd_op.unpool",
                                             "pd_op.correlation",
                                             "pd_op.depthwise_conv2d",
                                             "pd_op.grid_sample",
                                             "pd_op.shuffle_channel",
                                             "cf.yield",
                                             "pd_op.reshape",
                                             "pd_op.instance_norm",
                                             "pd_op.batch_norm_",
                                             "pd_op.sigmoid",
                                             "pd_op.bilinear_interp",
                                             "pd_op.multiply",
                                             "pd_op.shape",
                                             "pd_op.deformable_conv"};
  const std::set<std::string> op_with_axis_ = {
      "pd_op.all",
      "pd_op.amax",
      "pd_op.amin",
      "pd_op.any",
      //  "pd_op.argmax",
      "pd_op.argmin",
      "pd_op.argsort",
      "pd_op.box_coder",
      //  "pd_op.concat",
      "pd_op.cross",
      "pd_op.cross_entropy_with_softmax",
      "pd_op.cummax",
      "pd_op.cummin",
      "pd_op.cumsum",
      "pd_op.diagonal",
      "pd_op.fake_channel_wise_dequantize_max_abs",
      "pd_op.fake_channel_wise_quantize_abs_max",
      "pd_op.fake_channel_wise_quantize_dequantize_abs_max",
      "pd_op.flatten",
      "pd_op.flip",
      "pd_op.frame",
      "pd_op.frobenius_norm",
      "pd_op.gather",
      "pd_op.gumbel_softmax",
      "pd_op.index_add",
      "pd_op.index_select",
      "pd_op.index_select_strided",
      "pd_op.kthvalue",
      "pd_op.layer_norm",
      "pd_op.log_softmax",
      "pd_op.logcumsumexp",
      "pd_op.logsumexp",
      "pd_op.max",
      "pd_op.maxout",
      "pd_op.mean",
      "pd_op.mode",
      "pd_op.nanmedian",
      "pd_op.norm",
      "pd_op.overlap_add",
      "pd_op.p_norm",
      "pd_op.prod",
      "pd_op.put_along_axis",
      "pd_op.renorm",
      "pd_op.repeat_interleave",
      "pd_op.repeat_interleave_with_tensor_index",
      "pd_op.reverse",
      "pd_op.roll",
      "pd_op.slice",
      "pd_op.split",
      "pd_op.split_with_num",
      "pd_op.squeeze",
      "pd_op.stack",
      "pd_op.sum",
      "pd_op.take_along_axis",
      "pd_op.tensor_unfold",
      "pd_op.topk",
      "pd_op.trace",
      "pd_op.unbind",
      "pd_op.unique_consecutive",
      "pd_op.dequantize_linear",
      "pd_op.min",
      "pd_op.quantize_linear",
      "pd_op.softmax",
      "pd_op.sparse_momentum",
      "pd_op.unique",
      "pd_op.unsqueeze",
      "pd_op.unstack"};
};
}  // namespace
namespace pir {

std::unique_ptr<Pass> CreateAutoLayoutPass() {
  return std::make_unique<AutoLayoutPass>();
}

}  // namespace pir

REGISTER_IR_PASS(auto_layout_pass, AutoLayoutPass);
