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

#pragma once
#include <variant>
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/dll_decl.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/op_base.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/operation_utils.h"
#include "paddle/pir/dialect/shape/utils/shape_analysis.h"

namespace cinn {
namespace dialect {

class IR_API GroupOp : public pir::Op<GroupOp> {
 public:
  using Op::Op;
  static const char *name() { return "cinn_op.group"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    const std::vector<pir::Type> &output_types);

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    std::unique_ptr<pir::Block> &&block);

  static void Build(
      pir::Builder &builder,             // NOLINT
      pir::OperationArgument &argument,  // NOLINT
      const std::vector<pir::Type> &output_types,
      const std::unordered_map<
          ::pir::Operation *,
          std::vector<cinn::hlir::framework::pir::ScheduleInfoNode>>
          &alignment_schedule_info);

  pir::Block *block();
  std::vector<pir::Operation *> GetOperators();

  void VerifySig();
  void Print(pir::IrPrinter &printer);  // NOLINT
};

// FusionOp represents a subgraphs that can be fused to one kernel.
// Every GroupOp can be lowered to at least one FusionOp
class IR_API FusionOp : public pir::Op<FusionOp> {
 public:
  using Op::Op;
  static const char *name() { return "cinn_op.fusion"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    const std::vector<pir::Type> &output_types);

  pir::Block *block();
  std::vector<pir::Operation *> GetOperators();

  void VerifySig();
  void Print(pir::IrPrinter &printer);  // NOLINT
};

class IR_API ConcatOp
    : public pir::Op<ConcatOp, paddle::dialect::InferSymbolicShapeInterface> {
 public:
  using Op::Op;

  static const char *name() { return "cinn_op.concat"; }

  static constexpr uint32_t attributes_num = 1;

  static const char *attributes_name[attributes_num];

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    const std::vector<pir::Value> &inputs,
                    int axis);

  void VerifySig() const {}

  bool InferSymbolicShape(pir::ShapeConstraintIRAnalysis *shape_analysis);
};

class IR_API SplitOp : public pir::Op<SplitOp> {
 public:
  using Op::Op;

  static const char *name() { return "cinn_op.split"; }

  static constexpr uint32_t attributes_num = 2;

  static const char *attributes_name[attributes_num];

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value input,
                    const std::vector<int> &sections,
                    int axis);

  void VerifySig() const {}
};

class IR_API GenerateShapeOp
    : public pir::Op<GenerateShapeOp,
                     paddle::dialect::InferSymbolicShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "cinn_op.generate_shape"; }
  static constexpr uint32_t attributes_num = 2;
  static const char *attributes_name[attributes_num];

  struct SymbolBindingBase {
    std::string symbol_name;
    int64_t input_tensor_idx;
    int64_t input_tensor_dim_idx;
  };

  struct DataSymbolBinding : public SymbolBindingBase {};
  struct ShapeSymbolBinding : public SymbolBindingBase {};

  using SymbolBinding = std::variant<DataSymbolBinding, ShapeSymbolBinding>;

  using SymbolBindings = std::vector<SymbolBinding>;

  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    const std::vector<pir::Value> &inputs,
                    const std::vector<pir::Attribute> &output_dim_exprs,
                    const SymbolBindings &symbol_bindings);

  void VerifySig() {}

  pir::Value out() { return result(0); }

  bool InferSymbolicShape(pir::ShapeConstraintIRAnalysis *shape_analysis);

  static pir::Attribute ConvertSymbolBindingsToAttribute(
      pir::Builder &builder, const SymbolBindings &symbol_bindings);  // NOLINT
  static std::optional<SymbolBindings> ConvertAttributeToSymbolBindings(
      const pir::Attribute &symbol_bindings);
};

}  // namespace dialect
}  // namespace cinn

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(cinn::dialect::GroupOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(cinn::dialect::FusionOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(cinn::dialect::ConcatOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(cinn::dialect::SplitOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(cinn::dialect::GenerateShapeOp);
