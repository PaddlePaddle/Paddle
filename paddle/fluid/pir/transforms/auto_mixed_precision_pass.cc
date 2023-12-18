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

#include "paddle/fluid/pir/transforms/auto_mixed_precision_pass.h"
#include <memory>
#include <string>
#include <unordered_set>

#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/parameter.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

// This pattern is used to rewrite the CastOp that has a CastOp as its operand
class FoldMultiCastOpPattern
    : public pir::OpRewritePattern<paddle::dialect::CastOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::CastOp>::OpRewritePattern;

  bool MatchAndRewrite(
      paddle::dialect::CastOp cast_op,
      pir::PatternRewriter& rewriter) const override {  // NOLINT
    auto input_op = pir::GetDefiningOpForInput(cast_op, 0)
                        ->dyn_cast<paddle::dialect::CastOp>();
    if (!input_op) return false;
    auto op_type = pir::GetDataTypeFromValue(cast_op.out());
    auto new_cast_op = rewriter.Build<paddle::dialect::CastOp>(
        input_op.x().dyn_cast<pir::OpResult>(),
        paddle::dialect::TransToPhiDataType(op_type));
    rewriter.ReplaceOp(cast_op, std::vector<pir::Value>{new_cast_op.out()});
    rewriter.EraseOp(cast_op);
    return true;
  }
};

class AutoMixedPrecisionPattern : public pir::RewritePattern {
 public:
  AutoMixedPrecisionPattern(
      pir::IrContext* context,
      const phi::Place& place,
      const phi::DataType& precision_mode,
      bool enable_low_precision_io = false,
      pir::PatternBenefit benefit = 1,
      const std::vector<std::string>& generated_names = {})
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context, generated_names) {
    // precision_mode_ = precision_mode;  // should be set by user
    // place_ = place;                    // should be set by user
    // // enable_low_precision_io_ = enable_low_precision_io;
    // SetDefaultBlacklist();
    // SetDefaultWhitelist();
  }

  void SetDefaultBlacklist() {
    // black_list_.insert({
    //     paddle::dialect::ExpOp::name(),
    //     paddle::dialect::SquareOp::name(),
    //     paddle::dialect::LogOp::name(),
    //     // paddle::dialect::FetchOp::name(),

    //     // paddle::dialect::Mean::name(),
    //     // paddle::dialect::Sum::name(),
    //     paddle::dialect::SigmoidCrossEntropyWithLogitsOp::name(),
    // });
  }

  void SetDefaultWhitelist() {
    // white_list_.insert({paddle::dialect::FullOp::name(),
    //                     paddle::dialect::Conv2dOp::name(),
    //                     paddle::dialect::TransposeOp::name()});
    // return;
  }

  bool Match(pir::Operation* op) const override {
    // if enable_low_precision_io_ is true, all the op will be transformed into,
    // input and output included
    if (op->isa<pir::ParameterOp>() || op->isa<pir::SetParameterOp>() ||
        op->isa<paddle::dialect::CastOp>() ||
        op->isa<paddle::dialect::FullIntArrayOp>())
      return false;

    // if (!enable_low_precision_io_) {
    //   if (op->isa<paddle::dialect::FeedOp>()) return false;
    // }

    // if op is a full op, its user cannot be a scale op
    //   if (op->isa<paddle::dialect::FullOp>()) {
    //     auto use_ops = GetUseOpsForOutput(op, 0);
    //     for (auto [use_op, idx] : use_ops) {
    //       if (use_op->isa<paddle::dialect::ScaleOp>()) {
    //         return false;
    //       }
    //     }
    //   }

    //   if (!IsBuiltinOp(op)) {
    //     return OpHasFloatResult(op);
    //   }

    //   return true;
    // }
    return true;
  }
};

class AutoMixedPrecisionPass : public pir::Pass {
 public:
  AutoMixedPrecisionPass(const phi::Place& place,
                         const phi::DataType& precision_mode)
      : pir::Pass("auto_mixed_precision_pass", 1),
        place_(place),
        precision_mode_(precision_mode) {}

  bool Initialize(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<AutoMixedPrecisionPattern>(context, place_, precision_mode_);
    // ps.Add<FoldMultiCastOpPattern>(context);
    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation* op) override {
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 2;
    pir::ApplyPatternsGreedily(op->region(0), patterns_, cfg);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<::pir::ModuleOp>() && op->num_regions() > 0 &&
           place_ == paddle::PlaceType::kGPU &&
           (precision_mode_ == phi::DataType::FLOAT16 ||
            precision_mode_ == phi::DataType::BFLOAT16);
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
  phi::Place place_;
  phi::DataType precision_mode_;
  bool enable_low_precision_io_;

  std::unordered_set<std::string> black_list_;
  std::unordered_set<std::string> white_list_;

  std::unordered_set<pir::Operation*> op_run_low_precision_;

  void ProcessBlock(pir::Block* block) {}

  void GetOpPrecision(pir::Block* block) {
    for (auto& op_item : *block) {
      VLOG(6) << "op name " << op_item.name();
      auto op_name = op_item.name();
      bool support_low_precision = true;
      if (black_list.count(op_name)) {
        support_low_precision = false;
      } else if (IsBuiltinOp(&op_item)) {  // other builtin ops
        if (op->isa<pir::ParameterOp>() || op->isa<pir::SetParameterOp>())
          support_low_precision = false;
      } else if (op_item->isa<paddle::dialect::FeedOp>() ||
                 op_item->isa<paddle::dialect::FetchOp>()) {
        support_low_precision = enable_low_precision_io_;
      } else if (OpHasFloatResult(&op_item)) {  // pd op with float result
        auto op_type = op_name.substr(op_name.find(".") + 1);
        auto backend = ConvertPlaceToBackend(place_);
        support_low_precision =
            OpSupportPrecision(op_type, backend, precision_mode_);
      }
      if (support_low_precision) {
        op_run_low_precision_.insert(&op_item);
      }
    }
  }

  void UpdateOpPrecision(pir::Block* block) {
    for (auto& op_item : *block) {
      if (op_run_low_precision_.count(&op_item)) {
        RewriteOp(&op_item);
      }
    }
  }

  void RewriteOp(pir::Operation* op,
                 pir::PatternRewriter& rewriter) const {  // NOLINT
    LOG(INFO) << "Rewrite op " << op->name() << std::endl;
    if (IsBuiltinOp(op)) {
      RewriteBuiltinOp(op, rewriter);
      return;
    } else {
      RewritePdOp(op, rewriter);
      return;
    }
  }

  bool PhiKernelSupportPrecision(
      const std::string& op_type,
      phi::Backend backend,
      phi::DataType data_type,
      phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT) const {
    const auto& kernels = phi::KernelFactory::Instance().kernels();
    if (kernels.count(op_type) == 0) {
      return false;
    }
    phi::KernelKey kernel_key(backend, layout, data_type);
    return phi::KernelFactory::Instance().HasKernel(op_type, kernel_key);
  }

  phi::Backend ConvertPlaceToBackend(const phi::Place& place) const {
    switch (place.GetType()) {
      case phi::AllocationType::CPU:
        return phi::Backend::CPU;
      case phi::AllocationType::GPU:
        return phi::Backend::GPU;
      case phi::AllocationType::XPU:
        return phi::Backend::XPU;
      default:
        return phi::Backend::UNDEFINED;
    }
    return phi::Backend::UNDEFINED;
  }

  bool KernelSupportPrecision(
      const std::string& op_type,
      phi::Backend backend,
      phi::DataType precision,
      phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT) const {
    auto& phi_op_type = op_type;
    LOG(INFO) << "phi_op_type = " << phi_op_type << std::endl;

    bool support =
        PhiKernelSupportPrecision(phi_op_type, backend, precision, layout);
    if (backend == phi::Backend::GPU) {
      support |= PhiKernelSupportPrecision(
          phi_op_type, phi::Backend::GPUDNN, precision, layout);
    }

    if (!support) {
      const auto& all_kernels =
          paddle::framework::OperatorWithKernel::AllOpKernels();
      auto it = all_kernels.find(op_type);
      if (it != all_kernels.end()) {
        for (const auto& kern_pair : it->second) {
          if (ConvertPlaceToBackend(kern_pair.first.place_) == backend &&
              kern_pair.first.data_type_ ==
                  paddle::framework::TransToProtoVarType(precision)) {
            support = true;
            break;
          }
        }
      }
    }
    return support;
  }

  phi::Kernel GetPhiKernelInPrecision(const std::string& kernel_fn_str,
                                      phi::Backend backend,
                                      phi::DataType precision) const {
    if (backend == phi::Backend::GPU) {
      if (PhiKernelSupportPrecision(
              kernel_fn_str, phi::Backend::GPUDNN, precision)) {
        phi::KernelKey kernel_key(
            phi::Backend::GPUDNN, phi::DataLayout::ALL_LAYOUT, precision);
        return phi::KernelFactory::Instance().SelectKernel(kernel_fn_str,
                                                           kernel_key);
      }
      phi::KernelKey kernel_key(
          phi::Backend::GPU, phi::DataLayout::ALL_LAYOUT, precision);
      return phi::KernelFactory::Instance().SelectKernel(kernel_fn_str,
                                                         kernel_key);
    }
    return phi::KernelFactory::Instance().SelectKernel(
        kernel_fn_str,
        phi::KernelKey(backend, phi::DataLayout::ALL_LAYOUT, precision));
  }

  bool IsBuiltinOp(pir::Operation* op) const {
    return op->name().find("builtin") != std::string::npos;
  }

  bool OpSupportPrecision(const std::string& kernel_fn_str,
                          phi::Backend backend,
                          phi::DataType precision) const {
    // if the op is in white list, return true
    if (white_list_.count(kernel_fn_str)) {
      return true;
    }

    // if the op is in black list, return false
    if (black_list_.count(kernel_fn_str)) {
      return false;
    }

    return KernelSupportPrecision(kernel_fn_str, backend, precision);
  }

  bool ValueInPrecision(pir::Value value, phi::DataType precision) const {
    auto dtype = pir::GetDataTypeFromValue(value);
    return paddle::dialect::TransToPhiDataType(dtype) == precision;
  }

  void SetResultDataType(pir::Value result,
                         phi::DataType precision,
                         pir::IrContext* context) const {
    auto type = result.type();
    if (type.isa<paddle::dialect::DenseTensorType>()) {
      auto dense_type = type.dyn_cast<paddle::dialect::DenseTensorType>();
      auto new_type = paddle::dialect::DenseTensorType::get(
          context,
          paddle::dialect::TransToIrDataType(precision, context),
          dense_type.dims(),
          dense_type.data_layout(),
          dense_type.lod(),
          dense_type.offset());
      result.set_type(new_type);
    } else if (type.isa<pir::VectorType>()) {
      auto vec_type = type.dyn_cast<pir::VectorType>();
      auto output_num = vec_type.size();
      std::vector<pir::Type> results_type(output_num);
      for (size_t idx = 0; idx < output_num; ++idx) {
        auto dense_type =
            vec_type[idx].dyn_cast<paddle::dialect::DenseTensorType>();
        auto new_type = paddle::dialect::DenseTensorType::get(
            context,
            paddle::dialect::TransToIrDataType(precision, context),
            dense_type.dims(),
            dense_type.data_layout(),
            dense_type.lod(),
            dense_type.offset());
        results_type[idx] = new_type;
      }
      auto new_vec_type = pir::VectorType::get(context, results_type);
      result.set_type(new_vec_type);
    } else {
      LOG(INFO) << "result type is not DenseTensorType or VectorType"
                << std::endl;
    }
  }

  bool OpHasFloatResult(pir::Operation* op) const {
    for (size_t i = 0; i < op->num_results(); i++) {
      auto result = op->result(i);
      if (!result.type()) continue;
      if (result.type().isa<paddle::dialect::DenseTensorType>()) {
        auto dtype = pir::GetDataTypeFromValue(result);
        if (IsDataTypeFloat(paddle::dialect::TransToPhiDataType(dtype))) {
          return true;
        }
      } else if (result.type().isa<pir::VectorType>()) {
        auto vec_type = result.type().dyn_cast<pir::VectorType>();
        for (size_t j = 0; j < vec_type.size(); j++) {
          auto dtype =
              vec_type[j].dyn_cast<paddle::dialect::DenseTensorType>().dtype();
          if (IsDataTypeFloat(paddle::dialect::TransToPhiDataType(dtype))) {
            return true;
          }
        }
      }
    }
    LOG(INFO) << "op " << op->name() << " doesn't have float result"
              << std::endl;
    return false;
  }

  bool IsDataTypeFloat(const phi::DataType& dtype) const {
    return dtype == phi::DataType::FLOAT32 || dtype == phi::DataType::FLOAT16 ||
           dtype == phi::DataType::BFLOAT16;
  }

  phi::DataType OperandDataType(const pir::OpOperand& operand) const {
    auto dtype = pir::GetDataTypeFromValue(operand.source());
    return paddle::dialect::TransToPhiDataType(dtype);
  }

  bool IsOperandHasDenseTensorType(pir::OpOperand operand) const {
    return operand.type() &&
           operand.type().isa<paddle::dialect::DenseTensorType>();
  }

  void InsertCastOp(pir::Operation* op,
                    pir::OpOperand operand,
                    phi::DataType precision,
                    pir::PatternRewriter& rewriter) const {  // NOLINT
    auto value = operand.source();
    rewriter.set_insertion_point(op);  // before op
    paddle::dialect::CastOp cast_op =
        rewriter.Build<paddle::dialect::CastOp>(value, precision);
    operand.set_source(cast_op->result(0));
  }

  void RewriteBuiltinOp(pir::Operation* op,
                        pir::PatternRewriter& rewriter) const {  // NOLINT
    LOG(INFO) << "Rewrite builtin op " << op->name() << std::endl;
    // Rewrite CombineOp
    if (op->isa<pir::CombineOp>()) {
      // auto vec_type = op->result(0).type().dyn_cast<pir::VectorType>();
      auto input_num = op->num_operands();
      bool in_low_precision = false;
      bool should_insert_cast = false;
      for (size_t i = 0; i < input_num; ++i) {
        auto operand = op->operand(i);
        auto operand_dtype = OperandDataType(operand);
        if (operand_dtype == precision_mode_) {
          in_low_precision = true;
        } else if (IsDataTypeFloat(operand_dtype)) {
          should_insert_cast = true;
        }
      }
      if (in_low_precision && should_insert_cast) {
        LOG(INFO) << "Insert CastOp for CombineOp" << std::endl;
        for (size_t i = 0; i < input_num; ++i) {
          auto operand = op->operand(i);
          auto operand_dtype = OperandDataType(operand);
          if (operand_dtype != precision_mode_) {
            InsertCastOp(op, operand, precision_mode_, rewriter);
          }
        }
      }
      std::vector<pir::Type> inputs_type(input_num);
      for (size_t idx = 0; idx < input_num; ++idx) {
        inputs_type[idx] = op->operand(idx).type();
      }
      auto new_vec_type =
          pir::VectorType::get(rewriter.ir_context(), inputs_type);
      op->result(0).set_type(new_vec_type);
    }

    // Rewrite SliceOp
    if (op->isa<pir::SliceOp>()) {
      auto index =
          op->attribute("index").dyn_cast<pir::Int32Attribute>().data();
      auto input_type = op->operand(0).type().dyn_cast<pir::VectorType>();
      auto new_type = input_type[index];
      op->result(0).set_type(new_type);
    }

    // Rewrite SplitOp
    if (op->isa<pir::SplitOp>()) {
      auto input_type = op->operand(0).type().dyn_cast<pir::VectorType>();
      int output_num = op->num_results();
      for (int i = 0; i < output_num; ++i) {
        op->result(i).set_type(input_type[i]);
      }
    }
  }

  void RewritePdOp(pir::Operation* op,
                   pir::PatternRewriter& rewriter) const {  // NOLINT
    LOG(INFO) << "Rewrite pd op " << op->name() << std::endl;
    phi::Backend backend = ConvertPlaceToBackend(place_);
    std::string op_type = op->name().substr(op->name().find(".") + 1);

    // Rewrite FetchOp
    if (op->isa<paddle::dialect::FetchOp>()) {
      auto fetch_operand = op->operand(0);
      auto fetch_operand_dtype = OperandDataType(fetch_operand);
      if (enable_low_precision_io_) {
        SetResultDataType(
            op->result(0), precision_mode_, rewriter.ir_context());
      }
      if (!op->result(0).type().isa<paddle::dialect::DenseTensorType>()) return;
      auto result_dtype = paddle::dialect::TransToPhiDataType(
          pir::GetDataTypeFromValue(op->result(0)));
      if (fetch_operand_dtype != result_dtype) {
        LOG(INFO) << "Insert CastOp for FetchOp" << std::endl;
        InsertCastOp(op, fetch_operand, result_dtype, rewriter);
      }
      return;
    }
    // Rewrite FeedOp
    if (op->isa<paddle::dialect::FeedOp>() && enable_low_precision_io_) {
      SetResultDataType(op->result(0), precision_mode_, rewriter.ir_context());
      return;
    }

    // Rewrite ShareDataOp
    if (op->isa<paddle::dialect::ShareDataOp>()) {
      auto share_data_operand = op->operand(0);
      auto share_data_operand_dtype = OperandDataType(share_data_operand);
      SetResultDataType(
          op->result(0), share_data_operand_dtype, rewriter.ir_context());
      return;
    }

    if (OpSupportPrecision(op_type, backend, precision_mode_)) {
      // change result's dtype to low precision
      LOG(INFO) << "Change result's dtype to low precision " << op->name()
                << std::endl;

      if (op->HasAttribute("dtype") &&
          IsDataTypeFloat(
              op->attribute<paddle::dialect::DataTypeAttribute>("dtype")
                  .data())) {
        pir::Attribute attr_dtype = paddle::dialect::DataTypeAttribute::get(
            rewriter.ir_context(), precision_mode_);
        op->set_attribute("dtype", attr_dtype);
      }

      auto phi_kernel =
          GetPhiKernelInPrecision(op_type, backend, precision_mode_);
      PADDLE_ENFORCE(
          phi_kernel.IsValid(),
          phi::errors::PreconditionNotMet(
              "op [%s] kernel doesn't support precision [%s] on backend [%s]",
              op->name(),
              phi::DataTypeToString(precision_mode_).c_str(),
              paddle::experimental::BackendToString(backend).c_str()));

      auto args_def = phi_kernel.args_def();
      auto input_defs = args_def.input_defs();
      auto output_defs = args_def.output_defs();

      PADDLE_ENFORCE_EQ(
          op->num_results(),
          output_defs.size(),
          phi::errors::PreconditionNotMet(
              "op [%s] kernel output args defs should equal op outputs",
              op->name()));

      for (size_t i = 0; i < op->num_results(); i++) {
        auto result = op->result(i);
        if (!result.type()) continue;
        phi::DataType out_phi_dtype = output_defs[i].dtype;
        LOG(INFO) << "result dtype = " << phi::DataTypeToString(out_phi_dtype)
                  << std::endl;
        if (out_phi_dtype == phi::DataType::UNDEFINED)
          out_phi_dtype = precision_mode_;
        SetResultDataType(result, out_phi_dtype, rewriter.ir_context());
      }

      // if any of the op's input is not in low precision, insert cast op
      // input_defs will always be the smaller one?
      for (size_t i = 0; i < input_defs.size(); i++) {
        auto operand = op->operand(i);
        auto in_phi_dtype = input_defs[i].dtype;
        if (!IsOperandHasDenseTensorType(operand)) continue;
        auto operand_dtype = OperandDataType(operand);
        if (IsDataTypeFloat(operand_dtype) && operand_dtype != in_phi_dtype) {
          LOG(INFO) << "Support low precision, insert CastOp for " << op->name()
                    << " operand " << i << std::endl;
          InsertCastOp(op, operand, in_phi_dtype, rewriter);
        }
      }
    } else {  // current op doesn't support low precision, should cast to float
      // if the op's input is in low precision, insert cast op
      auto phi_dtype = phi::DataType::FLOAT32;
      for (size_t i = 0; i < op->num_operands(); i++) {
        auto operand = op->operand(i);
        if (!IsOperandHasDenseTensorType(operand)) continue;
        auto operand_dtype = OperandDataType(operand);
        if (IsDataTypeFloat(operand_dtype) &&
            operand_dtype == precision_mode_) {
          LOG(INFO) << "Not support low precision, insert CastOp for "
                    << op->name() << " operand " << i << std::endl;
          InsertCastOp(op, operand, phi_dtype, rewriter);
        }
      }
    }
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateAutoMixedPrecisionPass(
    const phi::Place& place, const phi::DataType& precision_mode) {
  return std::make_unique<AutoMixedPrecisionPass>(place, precision_mode);
}

}  // namespace pir
