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

#include "paddle/fluid/pir/transforms/general/auto_layout_insert_pass.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/common/enforce.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/pir/dialect/operator/interface/layout_transformation.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pass/utils.h"
#ifdef PADDLE_WITH_CINN
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
#endif

namespace {

extern const std::set<std::string> kOpsNchw;
extern const std::set<std::string> kOpsWithAxis;

class AutoLayoutInsertPass : public pir::Pass {
 public:
  AutoLayoutInsertPass() : pir::Pass("auto_layout_insert_pass", 2) {}
  AutoLayoutInsertPass(const std::set<std::string>& kOpsNhwc)  // NOLINT
      : pir::Pass("auto_layout_insert_pass", 2), kOpsNhwc_(kOpsNhwc) {}

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
  void RewriteLayout(pir::Operation* op) {
    // If op already register LayoutTransformationInterface, use it to rewrite.
    // if not, maybe it is a new operator. If it has a UnaryElementWiseTrait or
    // BinaryElementWiseTrait, then use input operands to rewrite the Layout.
    if (auto layout_interface =
            op->dyn_cast<paddle::dialect::LayoutTransformationInterface>()) {
      layout_interface.RewriteByLayout(op, common::DataLayout::NHWC);
    } else if (op->HasTrait<pir::UnaryElementWiseTrait>() ||
               op->HasTrait<pir::BinaryElementWiseTrait>()) {
      pir::TransLayoutCallbackFn callback = nullptr;
#ifdef PADDLE_WITH_CINN
      auto& shape_analysis =
          pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());
      callback = [&](pir::Value value, common::DataLayout new_layout) -> void {
        shape_analysis.UpdateShapeOrDataByTransLayout(
            value, pir::TransLayoutType::NCHW2NHWC);
      };
#endif
      for (size_t i = 0; i < op->results().size(); ++i) {
        op->result(i).set_type(op->operands_source().at(i).type());
        pir::SetNewLayoutForValue(
            op->result(i), common::DataLayout::NHWC, callback);
      }
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "`%s` should implement InferMetaInterface interface or rewrite "
          "manually, but not found.",
          op->name()));
    }
  }

  bool JudgeOperand(const pir::Value& operand,
                    const std::vector<int32_t>& layout) {
    if (operand.type().isa<pir::VectorType>()) {
      auto defined_op = operand.defining_op();
      for (auto inner_operand : defined_op->operands_source()) {
        if (JudgeOperand(inner_operand, kNchw2Nhwc_)) {
          return true;
        }
      }
      return false;
    } else {
      if (!JudgeValue(operand)) return false;
      auto transposeInputOp =
          operand.defining_op<paddle::dialect::TransposeOp>();
      if (!transposeInputOp) return false;
      pir::Operation* op = transposeInputOp.operation();
      if (!op->HasAttribute("source")) return false;
      auto source =
          transposeInputOp.attribute<pir::StrAttribute>("source").AsString();
      if (source != "auto_layout_pass") return false;
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
      is_insert_transpose = JudgeOperand(operand, kNhwc2Nchw_);
    }
    return is_insert_transpose;
  }

  // Convert NCHW permutation to NHWC permutation
  std::vector<pir::Attribute> ConvertNchwToNhwc(
      const std::vector<pir::Attribute>& nchw_perm) {
    std::vector<pir::Attribute> nhwc_perm(4);
    for (int i = 0; i < 4; ++i) {
      int32_t nchw_perm_value =
          nchw_perm[i].dyn_cast<pir::Int32Attribute>().data();
      int32_t new_value =
          nchw_perm_value == 0
              ? 0
              : (nchw_perm_value == 1 ? 3 : nchw_perm_value - 1);
      nhwc_perm[i] = pir::Int32Attribute::get(ctx_, new_value);
    }
    return nhwc_perm;
  }

  void TransformTransposePerm(paddle::dialect::TransposeOp* op) {
    std::vector<pir::Attribute> perm_values =
        op->attribute<pir::ArrayAttribute>("perm").AsVector();

    if (perm_values.size() == 4) {
      std::vector<pir::Attribute> new_perm_values =
          ConvertNchwToNhwc(perm_values);
      op->operation()->set_attribute(
          "perm", pir::ArrayAttribute::get(ctx_, new_perm_values));
    }
  }

  // return true if op is need to be skipped
  bool SkipUnsupportedOp(const pir::Operation& op) {
    // Skip special ops.
    if (op.operands().size() == 0) return true;
    if (op.HasTrait<pir::ImmutableLayoutTrait>()) return true;
    if (op.HasTrait<pir::UnaryElementWiseTrait>()) return false;
    if (op.HasTrait<pir::BinaryElementWiseTrait>()) {
      int32_t dim_size = -3;
      bool is_broadcast = false;
      int32_t inp_size = op.num_operands();
      for (int32_t i = 0; i < inp_size; ++i) {
        if (is_broadcast) break;
        auto type = op.operand_source(i)
                        .type()
                        .dyn_cast<paddle::dialect::DenseTensorType>();
        if (!type) {
          is_broadcast = true;
          break;
        }
        if (i == 0) {
          dim_size = type.dims().size();
        } else {
          if (dim_size != type.dims().size()) {
            is_broadcast = true;
            break;
          }
        }
      }
      return is_broadcast;
    }
    // if op not has UnaryElementWiseTrait and BinaryElementWiseTrait. And there
    // is no LayoutTransformationInterface, which means that this operator
    // cannot be taken over by AutoLayoutPass. We need to skip it.
    return !op.HasInterface<paddle::dialect::LayoutTransformationInterface>();
  }

  void TransferLayout(pir::Builder builder, pir::Block* block) {
    for (auto&& op_item : *block) {
      if (SkipUnsupportedOp(op_item)) continue;
      auto op = &op_item;
      auto op_name = op->name();

      // NHWC ops branch, Only support
      // conv2d、fused_conv2d_add_act、conv2d_transpose now, it will add white
      // list later.
      if (kOpsNhwc_.count(op_name)) {
        auto layout_interface =
            op->dyn_cast<paddle::dialect::LayoutTransformationInterface>();
        common::DataLayout new_layout = layout_interface.PreferLayout(op);
        if (new_layout != common::DataLayout::NHWC) continue;

        if (op->HasAttribute("data_format") &&
            op->attribute<pir::StrAttribute>("data_format").AsString() ==
                "NCHW") {
          VLOG(4) << "enter NHWC op: " << op_name;
          DoTransposeOpOperand(op, builder);
          RewriteLayout(op);
          DoTransposeOpResult(op, builder);
        }
      } else if (!kOpsNchw.count(op_name) && !kOpsWithAxis.count(op_name) &&
                 IsInsertTransposeOpBefore(op)) {
        // TODO(liujinnan): Remove the list and set the CanBeModified method for
        // all ops.
        if (op_name == "pd_op.pool2d" || op_name == "pd_op.reshape") {
          auto layout_interface =
              op->dyn_cast<paddle::dialect::LayoutTransformationInterface>();
          if (!layout_interface.CanBeModified(op)) continue;
        }
        VLOG(4) << "enter NCHW op: " << op_name;
        DoTransposeOpOperand(op, builder);
        if (auto transpose_op = op->dyn_cast<paddle::dialect::TransposeOp>()) {
          TransformTransposePerm(&transpose_op);
          continue;
        }
        RewriteLayout(op);
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
    auto InsertTranspose = [&](pir::OpOperand* operand) {
      // if operand defining op is reshape op, try to rewrite reshape op
      if (operand->source().defining_op<paddle::dialect::ReshapeOp>()) {
        auto reshape_op = operand->source()
                              .defining_op()
                              ->dyn_cast<paddle::dialect::ReshapeOp>();
        auto layout_interface =
            reshape_op
                ->dyn_cast<paddle::dialect::LayoutTransformationInterface>();
        if (layout_interface.CanBeModified(reshape_op)) {
          layout_interface.RewriteByLayout(reshape_op,
                                           common::DataLayout::NHWC);
          operand->set_source(reshape_op->result(0));
          return;
        }
      }
      auto transpose_op = builder.Build<paddle::dialect::TransposeOp>(
          operand->source(), kNchw2Nhwc_);
      transpose_op->set_attribute(
          "source",
          pir::StrAttribute::get(transpose_op->ir_context(),
                                 "auto_layout_pass"));
      pir::SetNewLayoutForValue(transpose_op->result(0),
                                common::DataLayout::NHWC);
      operand->set_source(transpose_op->result(0));
    };

    builder.set_insertion_point(op);
    // For conv2d, only transpose the input.
    if (op->isa<paddle::dialect::Conv2dOp>() ||
        op->isa<paddle::dialect::Conv2dTransposeOp>() ||
        op->isa<paddle::dialect::DepthwiseConv2dOp>()) {
      auto inp = op->operand(0);
      if (!JudgeValue(inp.source())) return;
      InsertTranspose(&inp);
      return;
    }

    for (auto& operand : op->operands()) {
      if (!JudgeValue(operand.source())) continue;
      // Can be optimize with cache when not eliminate the transpose op.
      InsertTranspose(&operand);
    }
  }
  void DoTransposeOpResult(pir::Operation* op,
                           pir::Builder& builder) {  // NOLINT
    builder.SetInsertionPointAfter(op);
    for (auto& result : op->results()) {
      if (!JudgeValue(result)) continue;
      auto transpose_op =
          builder.Build<paddle::dialect::TransposeOp>(result, kNhwc2Nchw_);
      transpose_op->set_attribute(
          "source",
          pir::StrAttribute::get(transpose_op->ir_context(),
                                 "auto_layout_pass"));
      pir::SetNewLayoutForValue(transpose_op->result(0),
                                common::DataLayout::NCHW);
      result.ReplaceAllUsesWith(transpose_op->result(0));
      transpose_op->operand(0).set_source(result);
    }
  }

  pir::IrContext* ctx_ = pir::IrContext::Instance();
  std::set<std::string> kOpsNhwc_;
  const std::vector<int32_t> kNchw2Nhwc_ = {0, 2, 3, 1};
  const std::vector<int32_t> kNhwc2Nchw_ = {0, 3, 1, 2};
};
const std::set<std::string> kOpsNchw = {"pd_op.max_pool2d_with_index",
                                        "pd_op.fractional_max_pool2d",
                                        "pd_op.unpool3d",
                                        "pd_op.unpool",
                                        // "pd_op.pool2d",
                                        "pd_op.correlation",
                                        "pd_op.depthwise_conv2d",
                                        "pd_op.grid_sample",
                                        "pd_op.shuffle_channel",
                                        "cf.yield",
                                        // "pd_op.reshape",
                                        "pd_op.instance_norm",
                                        // "pd_op.batch_norm_",
                                        "pd_op.bilinear_interp",
                                        "pd_op.shape",
                                        "pd_op.shape64",
                                        "pd_op.deformable_conv",
                                        "pd_op.set_value_with_tensor_",
                                        "pd_op.set_value_with_tensor"};
const std::set<std::string> kOpsWithAxis = {
    "pd_op.all",
    "pd_op.amax",
    "pd_op.amin",
    "pd_op.any",
    "pd_op.argmin",
    "pd_op.argsort",
    "pd_op.box_coder",
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

}  // namespace
namespace pir {

std::unique_ptr<Pass> CreateAutoLayoutInsertPass(
    const std::set<std::string>& kOpsNhwc) {
  return std::make_unique<AutoLayoutInsertPass>(kOpsNhwc);
}

}  // namespace pir

REGISTER_IR_PASS(auto_layout_insert_pass, AutoLayoutInsertPass);
