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
#include <unordered_map>
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

class AutoMixedPrecisionPass : public pir::Pass {
 public:
  AutoMixedPrecisionPass(const phi::Place& place,
                         const phi::DataType& precision_mode)
      : pir::Pass("auto_mixed_precision_pass", 1),
        place_(place),
        precision_mode_(precision_mode) {}

  bool Initialize(pir::IrContext* context) override {
    context_ = context;
    enable_low_precision_io_ = false;
    SetDefaultBlacklist();
    SetDefaultWhitelist();
    return true;
  }

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    for (size_t i = 0; i < op->num_regions(); ++i) {
      auto& region = op->region(i);
      for (auto& block : region) {
        LOG(INFO) << "===========Get Op Precision============" << std::endl;
        GetOpPrecision(block);
        LOG(INFO) << "===========Update Op Precision============" << std::endl;
        UpdateOpPrecision(block);

        LOG(INFO) << "===========" << op_run_low_precision_.size() << " of "
                  << block->size() << " ops"
                  << " run low precision" << std::endl;
        pir::Builder builder = pir::Builder(context_, block);
        LOG(INFO) << "===========Process Op Precision============" << std::endl;

        ProcessBlock(block, builder);
        LOG(INFO) << "===========Insert Cast Op Num : " << insert_cast_op_num_
                  << "============" << std::endl;
      }
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0 && place_ == paddle::PlaceType::kGPU &&
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
      VLOG(6) << "op name " << op->name();
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
      } else {  // pd op without float result
        support_low_precision = false;
        op_should_not_handle_.insert(op);
      }
      if (support_low_precision) {
        op_run_low_precision_.insert(op);
        LOG(INFO) << "op " << op->name() << " support low precision"
                  << std::endl;
      } else {
        LOG(INFO) << "op " << op->name() << " doesn't support low precision"
                  << std::endl;
      }
    }
  }

  bool CheckUseOpsScalaAttribute(
      const std::vector<std::pair<pir::Operation*, int32_t>>& use_ops) const {
    for (auto [use_op, idx] : use_ops) {
      if (use_op->HasInterface<paddle::dialect::OpYamlInfoInterface>()) {
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

  bool CheckOutputIsScalarAttribute(pir::Operation* op) {
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
        if (op->isa<paddle::dialect::CastOp>()) {  // add for cast op, not cast
                                                   // to float. i.e cast to bool
                                                   // or int
          // if datatype of result0 is not float, then cast op should be not
          // handled
          auto result_dtype = paddle::dialect::TransToPhiDataType(
              pir::GetDataTypeFromValue(op->result(0)));
          if (!IsPhiDataTypeFloat(result_dtype)) {
            op_run_low_precision_.erase(op);
            op_should_not_handle_.insert(op);
            precision_updated = true;
          }
        }
        if (!OpRunLowPrecision(op)) continue;
        if (CheckOutputIsScalarAttribute(op)) {  // Output is ScalarAttribute
          LOG(INFO) << "op " << op->name() << " output is ScalarAttribute"
                    << std::endl;
          op_run_low_precision_.erase(op);
          precision_updated = true;
        }
        if (!OpRunLowPrecision(op)) continue;
        for (size_t idx = 0; idx < op->num_operands(); ++idx) {
          if (!op->operand_source(idx)) continue;
          auto operand = op->operand(idx);
          if (operand.type() && operand.type().isa<pir::VectorType>()) {
            // check if there are all float in the vectortype
            auto vec_type = operand.type().dyn_cast<pir::VectorType>();
            if (IsVectorTypeFloat(vec_type)) {
              auto input_operation = GetDefiningOpForInput(op, idx);
              // 如果有一个是高精的话，则必须都跑在高精上
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

      // 输出的方式比较好
      // 一个op和他的输出是绑定的
      // op1 -> var1
      // var1 -> op2
      // var1 的精度

      // builtin.combine -> vector type
      // reshape op
      // reshape -> vector_type
    } while (precision_updated);
    // 产生(op1, op2)的跑在高精度
    // (op1 -> var1, op2 -> var2) => combine => var3 是 op3(属性输入)
    // print if op run low precision
    for (auto& op_item : *block) {
      auto op = &op_item;
      if (op_should_not_handle_.count(op)) {
        LOG(INFO) << "op " << op->name() << " should not be handled"
                  << std::endl;
      } else if (op_run_low_precision_.count(op)) {
        LOG(INFO) << "op " << op->name() << " run low precision" << std::endl;
      } else {
        LOG(INFO) << "op " << op->name() << " run high precision" << std::endl;
      }
    }
  }

  void RewriteOp(pir::Operation* op,
                 pir::Builder& builder) const {  // NOLINT
    LOG(INFO) << "Rewrite op " << op->name() << std::endl;
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
      } else if (result.type().isa<pir::VectorType>() &&
                 IsVectorTypeFloat(result.type().dyn_cast<pir::VectorType>())) {
      }
    }
    LOG(INFO) << "op " << op->name() << " doesn't have float result"
              << std::endl;
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
    LOG(INFO) << "Rewrite builtin op " << op->name() << std::endl;
    // Rewrite CombineOp
    if (op->isa<pir::CombineOp>()) {
      // auto vec_type = op->result(0).type().dyn_cast<pir::VectorType>();
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
                   pir::Builder& builder) const {  // NOLINT
    LOG(INFO) << "Rewrite pd op " << op->name() << std::endl;
    phi::Backend backend = ConvertPlaceToBackend(place_);
    std::string op_type = op->name().substr(op->name().find(".") + 1);

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
        LOG(INFO) << "Insert CastOp for FetchOp" << std::endl;
        DoInsertCastOp(op, fetch_operand, result_dtype, builder);
      }
      return;
    }
    // Rewrite FeedOp
    if (op->isa<paddle::dialect::FeedOp>() && OpRunLowPrecision(op)) {
      SetResultDataType(op->result(0), precision_mode_, builder.ir_context());
      return;
    }

    // Rewrite ShareDataOp
    if (op->isa<paddle::dialect::ShareDataOp>() && OpRunLowPrecision(op)) {
      SetResultDataType(op->result(0), precision_mode_, builder.ir_context());
      return;
    }

    // Other pd ops
    if (OpRunLowPrecision(op)) {
      // change result's dtype to low precision
      LOG(INFO) << "Change result's dtype to low precision " << op->name()
                << std::endl;

      if (op->HasAttribute("dtype") &&
          IsPhiDataTypeFloat(
              op->attribute<paddle::dialect::DataTypeAttribute>("dtype")
                  .data())) {
        pir::Attribute attr_dtype = paddle::dialect::DataTypeAttribute::get(
            builder.ir_context(), precision_mode_);
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
        if (!IsPhiDataTypeFloat(out_phi_dtype))
          continue;  // here handle op like "unequal", which has bool result
                     // type
        SetResultDataType(result, out_phi_dtype, builder.ir_context());
      }

      // if any of the op's input is not in low precision, insert cast op
      // input_defs will always be the smaller one?
      for (size_t i = 0; i < input_defs.size(); i++) {
        auto operand = op->operand(i);
        auto in_phi_dtype = input_defs[i].dtype;
        if (!IsOperandHasDenseTensorType(operand)) continue;
        auto operand_phi_dtype = GetPhiDataTypeFromOpOperand(operand);
        if (IsPhiDataTypeFloat(operand_phi_dtype) &&
            operand_phi_dtype != in_phi_dtype) {
          LOG(INFO) << "Support low precision, insert CastOp for " << op->name()
                    << " operand " << i << std::endl;
          DoInsertCastOp(op, operand, in_phi_dtype, builder);
        }
      }
    } else {  // current op doesn't support low precision, should cast to float
      // if the op's input is in low precision, insert cast op
      auto phi_dtype = phi::DataType::FLOAT32;
      for (size_t i = 0; i < op->num_operands(); i++) {
        auto operand = op->operand(i);
        if (!IsOperandHasDenseTensorType(operand)) continue;
        auto operand_phi_dtype = GetPhiDataTypeFromOpOperand(operand);
        if (IsPhiDataTypeFloat(operand_phi_dtype) &&
            operand_phi_dtype == precision_mode_) {
          LOG(INFO) << "Not support low precision, insert CastOp for "
                    << op->name() << " operand " << i << std::endl;
          DoInsertCastOp(op, operand, phi_dtype, builder);
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
