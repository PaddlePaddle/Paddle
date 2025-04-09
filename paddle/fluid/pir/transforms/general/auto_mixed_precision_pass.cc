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

#include "paddle/fluid/pir/transforms/general/auto_mixed_precision_pass.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/parameter.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

class AutoMixedPrecisionPass : public pir::Pass {
 public:
  AutoMixedPrecisionPass()
      : pir::Pass("auto_mixed_precision_pass", 1),
        place_(phi::CPUPlace{}),
        precision_mode_(phi::DataType::FLOAT16),
        enable_low_precision_io_(false),
        context_(nullptr),
        op_run_low_precision_(),
        op_should_not_handle_(),
        cached_cast_ops_() {}

  bool Initialize(pir::IrContext* context) override {
    PADDLE_ENFORCE_EQ(
        Has(pir::Pass::kPlaceAttr),
        true,
        common::errors::InvalidArgument(
            "Pass initialize failed."
            "When using AutoMixedPrecisionPass, place attribute is required!"
            "Use Set method to set the place attribute."));
    PADDLE_ENFORCE_EQ(
        Has("mixed_precision_mode"),
        true,
        common::errors::InvalidArgument(
            "Pass initialize failed."
            "When using AutoMixedPrecisionPass, precision_mode attribute is "
            "required!"
            "Use Set method to set the scope attribute."));

    PADDLE_ENFORCE_EQ(Has("enable_low_precision_io"),
                      true,
                      common::errors::InvalidArgument(
                          "Pass initialize failed."
                          "When using AutoMixedPrecisionPass, "
                          "enable_low_precision_io attribute is "
                          "required!"
                          "Use Set method to set the scope attribute."));

    PADDLE_ENFORCE_EQ(
        Has("mixed_black_list"),
        true,
        common::errors::InvalidArgument(
            "Pass initialize failed."
            "When using AutoMixedPrecisionPass, mixed_black_list attribute is "
            "required!"
            "Use Set method to set the scope attribute."));

    PADDLE_ENFORCE_EQ(
        Has("mixed_white_list"),
        true,
        common::errors::InvalidArgument(
            "Pass initialize failed."
            "When using AutoMixedPrecisionPass, mixed_white_list attribute is "
            "required!"
            "Use Set method to set the scope attribute."));

    place_ = Get<phi::Place>(pir::Pass::kPlaceAttr);
    precision_mode_ = Get<phi::DataType>("mixed_precision_mode");
    context_ = context;
    enable_low_precision_io_ = Get<bool>("enable_low_precision_io");
    black_list_ = Get<std::unordered_set<std::string>>("mixed_black_list");
    white_list_ = Get<std::unordered_set<std::string>>("mixed_white_list");
    SetDefaultBlacklist();
    return true;
  }

  void Run(pir::Operation* op) override {
    SubBlockRun(op->GetParentProgram()->block());
    AddStatistics(op_run_low_precision_.size());
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0 && op->isa<pir::ModuleOp>() &&
           place_ == paddle::PlaceType::kGPU &&
           (precision_mode_ == phi::DataType::FLOAT16 ||
            precision_mode_ == phi::DataType::BFLOAT16);
  }

 private:
  phi::Place place_;
  phi::DataType precision_mode_;
  bool enable_low_precision_io_;
  pir::IrContext* context_;

  std::unordered_set<std::string> black_list_;
  std::unordered_set<std::string> white_list_;

  std::unordered_set<pir::Operation*> op_run_low_precision_;
  std::unordered_set<pir::Operation*> op_should_not_handle_;
  std::unordered_map<pir::Value, paddle::dialect::CastOp> cached_cast_ops_;

  int insert_cast_op_num_ = 0;

  void SetDefaultBlacklist() {
    black_list_.insert({
        paddle::dialect::ExpOp::name(),
        paddle::dialect::SquareOp::name(),
        paddle::dialect::LogOp::name(),
        paddle::dialect::MeanOp::name(),
        paddle::dialect::SumOp::name(),
        paddle::dialect::SigmoidCrossEntropyWithLogitsOp::name(),
        paddle::dialect::CrossEntropyWithSoftmax_Op::name(),
        paddle::dialect::ArrayToTensorOp::name(),
    });
  }

  void ProcessBlock(pir::Block* block, pir::Builder& builder) {  // NOLINT
    for (auto& op_item : *block) {
      auto op = &op_item;
      if (op_should_not_handle_.count(op)) continue;
      RewriteOp(op, builder);
    }
  }

  void GetOpPrecision(pir::Block* block) {
    for (auto& op_item : *block) {
      auto op = &op_item;
      auto op_name = op->name();
      bool support_low_precision = true;
      if (black_list_.count(op_name)) {
        support_low_precision = false;
      } else if (IsBuiltinOp(op)) {  // other builtin ops
        if (op->isa<pir::ParameterOp>() || op->isa<pir::SetParameterOp>())
          support_low_precision = false;
      } else if (op->isa<paddle::dialect::FeedOp>() ||
                 op->isa<paddle::dialect::FetchOp>()) {
        support_low_precision = enable_low_precision_io_;
      } else if (OpHasFloatOpOperand(op) ||
                 OpHasFloatResult(op)) {  // pd op without float result,
        auto op_type = op_name.substr(op_name.find(".") + 1);
        auto backend = ConvertPlaceToBackend(place_);
        support_low_precision =
            OpSupportPrecision(op_type, backend, precision_mode_);
        if (op->isa<paddle::dialect::ScaleOp>() && !OpHasFloatResult(op)) {
          support_low_precision = false;
          op_should_not_handle_.insert(op);
        }
      } else {  // pd op without float result
        support_low_precision = false;
        op_should_not_handle_.insert(op);
      }
      if (support_low_precision) {
        op_run_low_precision_.insert(op);
      }
    }
  }

  bool CheckUseOpsScalaAttribute(
      const std::vector<std::pair<pir::Operation*, int32_t>>& use_ops) const {
    for (auto [use_op, idx] : use_ops) {
      if (use_op->isa<pir::CombineOp>()) {
        if (CheckOutputIsScalarAttribute(use_op)) {
          return true;
        }
      } else if (use_op->HasInterface<paddle::dialect::OpYamlInfoInterface>()) {
        auto [input_infos, _1, _2, _3, _4] =
            use_op->dyn_cast<paddle::dialect::OpYamlInfoInterface>()
                .GetOpInfo();
        if (input_infos[idx].type_name.find("ScalarAttribute") !=
            std::string::npos) {
          return true;
        }
      }
    }
    return false;
  }

  bool CheckOutputIsScalarAttribute(pir::Operation* op) const {
    for (uint32_t i = 0; i < op->num_results(); i++) {
      auto use_ops = pir::GetUseOpsForOutput(op, i);
      if (CheckUseOpsScalaAttribute(use_ops)) return true;
    }
    return false;
  }

  void UpdateOpPrecision(pir::Block* block) {
    bool precision_updated = false;
    do {
      precision_updated = false;
      // handle full like op
      for (auto& op_item : *block) {
        auto op = &op_item;
        if (op_should_not_handle_.count(op)) continue;
        if (!OpRunLowPrecision(op)) continue;
        if (op->isa<paddle::dialect::FullLikeOp>()) {
          auto input_operation = GetDefiningOpForInput(op, 0);
          if (!op_run_low_precision_.count(input_operation)) {
            op_run_low_precision_.erase(op);
            precision_updated = true;
          }
        }
        if (!OpRunLowPrecision(op)) continue;
        // if datatype of cast op result is not float, then cast op should be
        // not handled
        if (op->isa<paddle::dialect::CastOp>()) {
          auto result_dtype = paddle::dialect::TransToPhiDataType(
              pir::GetDataTypeFromValue(op->result(0)));
          if (!IsPhiDataTypeFloat(result_dtype)) {
            op_run_low_precision_.erase(op);
            op_should_not_handle_.insert(op);
            precision_updated = true;
          }
        }
        if (!OpRunLowPrecision(op)) continue;
        // if consumer's input is a ScalarAttribute, the producer should be in
        // high precision
        if (CheckOutputIsScalarAttribute(op)) {
          op_run_low_precision_.erase(op);
          precision_updated = true;
        }
        if (!OpRunLowPrecision(op)) continue;
        // if the producer's output is in float VectorType, then the precision
        // between two op should be the same
        for (size_t idx = 0; idx < op->num_operands(); ++idx) {
          if (!op->operand_source(idx)) continue;
          auto operand = op->operand(idx);
          if (operand.type() && operand.type().isa<pir::VectorType>()) {
            // check if there are all float in the vector type
            auto vec_type = operand.type().dyn_cast<pir::VectorType>();
            if (IsVectorTypeFloat(vec_type)) {
              auto input_operation = GetDefiningOpForInput(op, idx);
              if (!op_run_low_precision_.count(op) ||
                  !op_run_low_precision_.count(input_operation)) {
                op_run_low_precision_.erase(op);
                op_run_low_precision_.erase(input_operation);
                precision_updated = true;
              }
            }
          }
        }
      }
    } while (precision_updated);
  }

  void RewriteOp(pir::Operation* op,
                 pir::Builder& builder) {  // NOLINT
    if (IsBuiltinOp(op)) {
      RewriteBuiltinOp(op, builder);
      return;
    } else {
      RewritePdOp(op, builder);
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
      case phi::AllocationType::CUSTOM:
        return phi::Backend::CUSTOM;
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

  void SetResultDataType(pir::Value result,
                         phi::DataType precision,
                         pir::IrContext* context) const {
    auto type = result.type();
    if (type.isa<paddle::dialect::DenseTensorType>()) {
      auto dense_type = type.dyn_cast<paddle::dialect::DenseTensorType>();
      if (IsDenseTensorTypeFloat(dense_type)) {
        auto new_type = paddle::dialect::DenseTensorType::get(
            context,
            paddle::dialect::TransToIrDataType(precision, context),
            dense_type.dims(),
            dense_type.data_layout(),
            dense_type.lod(),
            dense_type.offset());
        result.set_type(new_type);
      }
    } else if (type.isa<pir::VectorType>()) {
      auto vec_type = type.dyn_cast<pir::VectorType>();
      auto output_num = vec_type.size();
      std::vector<pir::Type> results_type(output_num);
      for (size_t idx = 0; idx < output_num; ++idx) {
        auto dense_type =
            vec_type[idx].dyn_cast<paddle::dialect::DenseTensorType>();
        if (IsDenseTensorTypeFloat(dense_type)) {
          auto new_type = paddle::dialect::DenseTensorType::get(
              context,
              paddle::dialect::TransToIrDataType(precision, context),
              dense_type.dims(),
              dense_type.data_layout(),
              dense_type.lod(),
              dense_type.offset());
          results_type[idx] = new_type;
        } else {
          results_type[idx] = dense_type;
        }
      }
      auto new_vec_type = pir::VectorType::get(context, results_type);
      result.set_type(new_vec_type);
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "result type is not DenseTensorType or VectorType"));
    }
  }

  bool OpHasFloatOpOperand(pir::Operation* op) const {
    for (size_t i = 0; i < op->num_operands(); i++) {
      auto operand = op->operand_source(i);
      if (!operand.type()) continue;
      if (operand.type().isa<paddle::dialect::DenseTensorType>() &&
          IsDenseTensorTypeFloat(
              operand.type().dyn_cast<paddle::dialect::DenseTensorType>())) {
        return true;
      } else if (operand.type().isa<pir::VectorType>() &&
                 IsVectorTypeFloat(
                     operand.type().dyn_cast<pir::VectorType>())) {
        return true;
      }
    }
    return false;
  }

  bool OpHasFloatResult(pir::Operation* op) const {
    for (size_t i = 0; i < op->num_results(); i++) {
      auto result = op->result(i);
      if (!result.type()) continue;
      if (result.type().isa<paddle::dialect::DenseTensorType>() &&
          IsDenseTensorTypeFloat(
              result.type().dyn_cast<paddle::dialect::DenseTensorType>())) {
        return true;
      } else if (result.type().isa<pir::VectorType>() &&
                 IsVectorTypeFloat(result.type().dyn_cast<pir::VectorType>())) {
        return true;
      }
    }
    return false;
  }

  bool IsPhiDataTypeFloat(const phi::DataType& dtype) const {
    return dtype == phi::DataType::FLOAT32 || dtype == phi::DataType::FLOAT16 ||
           dtype == phi::DataType::BFLOAT16;
  }

  bool IsDenseTensorTypeFloat(
      paddle::dialect::DenseTensorType dense_type) const {
    auto dtype = dense_type.dtype();
    return IsPhiDataTypeFloat(paddle::dialect::TransToPhiDataType(dtype));
  }

  bool IsVectorTypeFloat(pir::VectorType vec_type) const {
    size_t output_num = vec_type.size();
    for (size_t j = 0; j < output_num; j++) {
      auto dtype =
          vec_type[j].dyn_cast<paddle::dialect::DenseTensorType>().dtype();
      if (!IsPhiDataTypeFloat(paddle::dialect::TransToPhiDataType(dtype))) {
        return false;
      }
    }
    return true;
  }

  phi::DataType GetPhiDataTypeFromOpOperand(
      const pir::OpOperand& operand) const {
    return GetPhiDataTypeFromValue(operand.source());
  }

  phi::DataType GetPhiDataTypeFromValue(const pir::Value& value) const {
    auto dtype = pir::GetDataTypeFromValue(value);
    return paddle::dialect::TransToPhiDataType(dtype);
  }

  bool IsOperandHasDenseTensorType(pir::OpOperand operand) const {
    return operand.type() &&
           operand.type().isa<paddle::dialect::DenseTensorType>();
  }
  bool IsOperandHasDenseTensorVectorType(pir::OpOperand operand) const {
    return operand.type() && operand.type().isa<pir::VectorType>();
  }

  void DoInsertCastOp(pir::Operation* op,
                      pir::OpOperand operand,
                      phi::DataType precision,
                      pir::Builder& builder) {  // NOLINT
    auto value = operand.source();
    if (cached_cast_ops_.count(value)) {
      operand.set_source(cached_cast_ops_[value]->result(0));
      return;
    }
    builder.set_insertion_point(op);  // before op
    paddle::dialect::CastOp cast_op =
        builder.Build<paddle::dialect::CastOp>(value, precision);
    operand.set_source(cast_op->result(0));
    cached_cast_ops_[value] = cast_op;
    insert_cast_op_num_++;
  }

  bool OpRunLowPrecision(pir::Operation* op) const {
    return op_run_low_precision_.count(op);
  }

  void RewriteBuiltinOp(pir::Operation* op,
                        pir::Builder& builder) {  // NOLINT
    // Rewrite CombineOp
    if (op->isa<pir::CombineOp>()) {
      auto input_num = op->num_operands();
      if (OpRunLowPrecision(op)) {
        for (size_t i = 0; i < input_num; ++i) {
          auto operand = op->operand(i);
          auto operand_phi_dtype = GetPhiDataTypeFromOpOperand(operand);
          if (IsPhiDataTypeFloat(operand_phi_dtype) &&
              operand_phi_dtype != precision_mode_) {
            DoInsertCastOp(op, operand, precision_mode_, builder);
          }
        }
        std::vector<pir::Type> inputs_type(input_num);
        for (size_t idx = 0; idx < input_num; ++idx) {
          inputs_type[idx] = op->operand(idx).type();
        }
        auto new_vec_type =
            pir::VectorType::get(builder.ir_context(), inputs_type);
        op->result(0).set_type(new_vec_type);
      } else {
        for (size_t i = 0; i < input_num; ++i) {
          auto operand = op->operand(i);
          auto operand_phi_dtype = GetPhiDataTypeFromOpOperand(operand);
          if (operand_phi_dtype == precision_mode_) {
            DoInsertCastOp(op, operand, phi::DataType::FLOAT32, builder);
          }
        }
      }
    }

    // Rewrite SliceOp
    if (op->isa<pir::SliceOp>()) {
      if (!OpRunLowPrecision(op)) return;
      auto index =
          op->attribute("index").dyn_cast<pir::Int32Attribute>().data();
      auto input_type = op->operand(0).type().dyn_cast<pir::VectorType>();
      auto new_type = input_type[index];
      op->result(0).set_type(new_type);
    }

    // Rewrite SplitOp
    if (op->isa<pir::SplitOp>()) {
      if (!OpRunLowPrecision(op)) return;
      auto input_type = op->operand(0).type().dyn_cast<pir::VectorType>();
      int output_num = op->num_results();
      for (int i = 0; i < output_num; ++i) {
        op->result(i).set_type(input_type[i]);
      }
    }
  }

  void RewritePdOp(pir::Operation* op,
                   pir::Builder& builder) {  // NOLINT
    std::string op_type = op->name().substr(op->name().find(".") + 1);
    phi::Backend backend = ConvertPlaceToBackend(place_);
    // Rewrite FetchOp
    if (op->isa<paddle::dialect::FetchOp>()) {
      auto fetch_operand = op->operand(0);
      auto fetch_operand_phi_dtype = GetPhiDataTypeFromOpOperand(fetch_operand);
      if (OpRunLowPrecision(op)) {
        SetResultDataType(op->result(0), precision_mode_, builder.ir_context());
      }
      if (!op->result(0).type().isa<paddle::dialect::DenseTensorType>()) return;
      auto result_dtype = paddle::dialect::TransToPhiDataType(
          pir::GetDataTypeFromValue(op->result(0)));
      if (fetch_operand_phi_dtype != result_dtype) {
        DoInsertCastOp(op, fetch_operand, result_dtype, builder);
      }
      return;
    }
    // Rewrite FeedOp
    if (op->isa<paddle::dialect::FeedOp>() && OpRunLowPrecision(op)) {
      SetResultDataType(op->result(0), precision_mode_, builder.ir_context());
      return;
    }

    // Rewrite ShareData_Op
    if (op->isa<paddle::dialect::ShareData_Op>() && OpRunLowPrecision(op)) {
      SetResultDataType(op->result(0), precision_mode_, builder.ir_context());
      return;
    }
    // Other pd ops
    if (OpRunLowPrecision(op)) {
      auto phi_kernel =
          GetPhiKernelInPrecision(op_type, backend, precision_mode_);
      PADDLE_ENFORCE(
          phi_kernel.IsValid(),
          common::errors::PreconditionNotMet(
              "op [%s] kernel doesn't support precision [%s] on backend [%s]",
              op->name(),
              phi::DataTypeToString(precision_mode_).c_str(),
              paddle::experimental::BackendToString(backend).c_str()));

      auto args_def = phi_kernel.args_def();
      auto input_defs = args_def.input_defs();
      auto output_defs = args_def.output_defs();

      // if any of the op's input is not in low precision, insert cast op
      for (size_t i = 0; i < input_defs.size(); i++) {
        auto operand = op->operand(i);
        auto in_phi_dtype = input_defs[i].dtype;
        if (!IsOperandHasDenseTensorType(operand)) continue;
        auto operand_phi_dtype = GetPhiDataTypeFromOpOperand(operand);

        if (!InputVarsNotConvert(op, i) &&
            IsPhiDataTypeFloat(operand_phi_dtype) &&
            operand_phi_dtype != in_phi_dtype) {
          DoInsertCastOp(op, operand, in_phi_dtype, builder);
        }
      }

      // change result's dtype to low precision
      if (op->HasAttribute("dtype")) {
        auto phi_dtype = op->attribute("dtype")
                             .dyn_cast<paddle::dialect::DataTypeAttribute>()
                             .data();
        if (IsPhiDataTypeFloat(phi_dtype)) {
          pir::Attribute attr_dtype = paddle::dialect::DataTypeAttribute::get(
              builder.ir_context(), precision_mode_);
          op->set_attribute("dtype", attr_dtype);
        } else if (phi_dtype ==
                   phi::DataType::UNDEFINED) {  // dtype is not set, means all
                                                // ok
          pir::Attribute attr_dtype = paddle::dialect::DataTypeAttribute::get(
              builder.ir_context(), precision_mode_);
          op->set_attribute("dtype", attr_dtype);
        } else {
          return;  // don't modify output dtype
        }
      }

      PADDLE_ENFORCE_EQ(
          op->num_results(),
          output_defs.size(),
          common::errors::PreconditionNotMet(
              "op [%s] kernel output args defs should equal op outputs",
              op->name()));

      for (size_t i = 0; i < op->num_results(); i++) {
        auto result = op->result(i);
        if (!result.type()) continue;
        phi::DataType out_phi_dtype = output_defs[i].dtype;
        if (out_phi_dtype == phi::DataType::UNDEFINED)
          out_phi_dtype = precision_mode_;
        if (!IsPhiDataTypeFloat(out_phi_dtype))
          continue;  // here handle op like "unequal", which has bool result
                     // type

        if (!OutputVarsNotConvert(op, i)) {
          SetResultDataType(result, out_phi_dtype, builder.ir_context());
        }
      }
    } else {
      // current op doesn't support low precision
      // if the op's input is in low precision, insert cast op
      auto phi_dtype = phi::DataType::FLOAT32;
      for (size_t i = 0; i < op->num_operands(); i++) {
        auto operand = op->operand(i);
        if (IsOperandHasDenseTensorType(operand)) {
          auto operand_phi_dtype = GetPhiDataTypeFromOpOperand(operand);
          if (IsPhiDataTypeFloat(operand_phi_dtype) &&
              operand_phi_dtype == precision_mode_) {
            if (IsFakeFullOperandWhileOp(op, operand)) {
              SetFakeFullDataType(operand, builder);
              continue;
            }
            DoInsertCastOp(op, operand, phi_dtype, builder);
          }
        } else if (IsOperandHasDenseTensorVectorType(operand)) {
          auto defining_op_ = operand.source().defining_op();
          if (defining_op_->isa<pir::CombineOp>()) {
            auto input_num = defining_op_->num_operands();
            for (size_t i = 0; i < input_num; ++i) {
              auto operand = defining_op_->operand(i);
              auto operand_phi_dtype = GetPhiDataTypeFromOpOperand(operand);
              if (IsPhiDataTypeFloat(operand_phi_dtype) &&
                  operand_phi_dtype != phi::DataType::FLOAT32) {
                DoInsertCastOp(
                    defining_op_, operand, phi::DataType::FLOAT32, builder);
              }
            }
            std::vector<pir::Type> inputs_type(input_num);
            for (size_t idx = 0; idx < input_num; ++idx) {
              inputs_type[idx] = defining_op_->operand(idx).type();
            }
            auto new_vec_type =
                pir::VectorType::get(builder.ir_context(), inputs_type);
            defining_op_->result(0).set_type(new_vec_type);
          }
        } else {
          continue;
        }
      }
    }
  }

  std::vector<int64_t> ConvertDDimToVector(const common::DDim& ddim) {
    std::vector<int64_t> dims;
    for (int i = 0; i < ddim.size(); ++i) {
      dims.push_back(ddim[i]);
    }
    return dims;
  }

  bool IsFakeFullOp(pir::OpOperand operand) {
    auto defining_op_ = operand.source().defining_op();
    if (defining_op_->isa<paddle::dialect::FullOp>()) {
      auto full_op = defining_op_->dyn_cast<paddle::dialect::FullOp>();
      auto shape_attr = full_op.attribute("shape")
                            .dyn_cast<paddle::dialect::IntArrayAttribute>();
      auto shape_dims = shape_attr.data().GetData();
      auto result_dims = full_op.out()
                             .type()
                             .dyn_cast<paddle::dialect::DenseTensorType>()
                             .dims();
      std::vector<int64_t> result_dims_vec = ConvertDDimToVector(result_dims);

      if (shape_dims.size() != result_dims_vec.size()) {
        return true;
      }
      for (size_t i = 0; i < shape_dims.size(); ++i) {
        if (shape_dims[i] != result_dims_vec[i]) {
          return true;
        }
      }
      return false;
    }
    return false;
  }

  bool IsFakeFullOperandWhileOp(pir::Operation* op, pir::OpOperand operand) {
    return op->isa<paddle::dialect::WhileOp>() && IsFakeFullOp(operand);
  }

  void SetFakeFullDataType(pir::OpOperand operand,
                           pir::Builder& builder) {  // NOLINT
    auto defining_op_ = operand.source().defining_op();
    auto full_op = defining_op_->dyn_cast<paddle::dialect::FullOp>();
    pir::Attribute attr_dtype = paddle::dialect::DataTypeAttribute::get(
        builder.ir_context(), phi::DataType::FLOAT32);
    full_op->set_attribute("dtype", attr_dtype);

    SetResultDataType(
        full_op.out(), phi::DataType::FLOAT32, builder.ir_context());
  }

  void SubOpRun(pir::Operation* op) {
    for (auto& region : *op) {
      for (auto& block : region) {
        SubBlockRun(&block);  // subblock
      }
    }
  }

  void SubBlockRun(pir::Block* block) {
    GetOpPrecision(block);
    UpdateOpPrecision(block);
    pir::Builder builder = pir::Builder(context_, block);
    ProcessBlock(block, builder);
    cached_cast_ops_.clear();
    op_should_not_handle_.clear();
    for (auto& op : *block) {
      SubOpRun(&op);
    }
  }

  bool InputVarsNotConvert(pir::Operation* op, size_t index) {
    const std::unordered_map<std::string,
                             std::vector<std::pair<size_t, std::string>>>
        op_input_indices = {
            {"pd_op.batch_norm",
             {{1, "mean"}, {2, "variance"}, {3, "scale"}, {4, "bias"}}},
            {"pd_op.instance_norm", {{1, "scale"}, {2, "bias"}}},
            {"pd_op.layer_norm", {{1, "scale"}, {2, "bias"}}},
            {"pd_op.fused_multi_transformer",
             {{1, "ln_scales"},
              {2, "ln_biases"},
              {14, "ffn_ln_scales"},
              {15, "ffn_ln_biases"}}},
            {"pd_op.fused_bias_dropout_residual_layer_norm",
             {{3, "ln_scale"}, {4, "ln_bias"}}},
            {"pd_op.quantize_linear", {{1, "scale"}, {2, "zero_point"}}},
            {"pd_op.dequantize_linear", {{1, "scale"}, {2, "zero_point"}}}};

    auto it = op_input_indices.find(op->name());
    if (it != op_input_indices.end()) {
      const auto& index_pairs = it->second;
      for (const auto& [idx, param_name] : index_pairs) {
        if (idx == index) {
          VLOG(5) << "Skipping input conversion for operation: " << op->name()
                  << ", parameter index: " << index
                  << ", parameter name: " << param_name;
          return true;
        }
      }
    }

    return false;
  }

  bool OutputVarsNotConvert(pir::Operation* op, size_t index) {
    const std::unordered_map<std::string,
                             std::vector<std::pair<size_t, std::string>>>
        op_output_indices = {
            {"pd_op.batch_norm",
             {{1, "mean_out"},
              {2, "variance_out"},
              {3, "saved_mean"},
              {4, "saved_variance"}}},
            {"pd_op.layer_norm", {{1, "mean"}, {2, "variance"}}},
            {"pd_op.instance_norm", {{1, "mean"}, {2, "variance"}}}};

    auto it = op_output_indices.find(op->name());
    if (it != op_output_indices.end()) {
      const auto& index_pairs = it->second;
      for (const auto& [idx, param_name] : index_pairs) {
        if (idx == index) {
          VLOG(5) << "Skipping output conversion for operation: " << op->name()
                  << ", parameter index: " << index
                  << ", parameter name: " << param_name;
          return true;
        }
      }
    }

    return false;
  }
};
}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateAutoMixedPrecisionPass() {
  return std::make_unique<AutoMixedPrecisionPass>();
}

}  // namespace pir

REGISTER_IR_PASS(auto_mixed_precision_pass, AutoMixedPrecisionPass);
