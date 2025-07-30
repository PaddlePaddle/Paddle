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

#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/op_trait.h"

namespace pir {

class Program;
class Block;
constexpr char kStopGradientAttrName[] = "stop_gradient";
constexpr char kOutputDimExprs[] = "output_dim_exprs";
constexpr char kSymbolBindings[] = "symbol_bindings";
///
/// \brief ModuleOp
///
class IR_API ModuleOp : public pir::Op<ModuleOp> {
 public:
  using Op::Op;
  static const char *name() { return "builtin.module"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  void VerifySig() const;
  Program *program();
  Block &block();

  //
  // As the top operation, ModuleOp only support create&destroy through
  // below interface: "create"&"destroy".
  static ModuleOp Create(IrContext *context, Program *pointer);
  void Destroy();
};

class IR_API GroupOp : public pir::Op<GroupOp> {
 public:
  using Op::Op;
  static const char *name() { return "builtin.group"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    const std::vector<pir::Type> &output_types);

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    std::unique_ptr<Block> &&block);

  Block *block();
  Block *block() const;
  std::vector<pir::Operation *> GetOperators() const;
  void VerifySig();
  void Print(pir::IrPrinter &printer);  // NOLINT
};

///
/// \brief ParameterOp: OpResult = ParameterOp({StrAttribute,
/// StrAttribute})
///
class IR_API ParameterOp : public pir::Op<ParameterOp> {
 public:
  using Op::Op;
  static const char *name() { return "builtin.parameter"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    const std::string &name,
                    Type type);
  void VerifySig() const;
  std::string param_name() const;

 private:
  static void PassStopGradients(OperationArgument &argument);  // NOLINT
};

///
/// \brief SetParameterOp: SetParameterOp(OpOperand, {StrAttribute,
/// StrAttribute})
///
class IR_API SetParameterOp : public pir::Op<SetParameterOp, SideEffectTrait> {
 public:
  using Op::Op;
  static const char *name() { return "builtin.set_parameter"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value parameter,
                    const std::string &name);
  void VerifySig() const;
};

///
/// \brief ShadowOutputOp: ShadowOutputOp(OpOperand, {StrAttribute,
/// StrAttribute})
///
class IR_API ShadowOutputOp
    : public pir::Op<ShadowOutputOp, SideEffectTrait, ImmutableLayoutTrait> {
 public:
  using Op::Op;
  static const char *name() { return "builtin.shadow_output"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value parameter,
                    const std::string &name);
  void VerifySig() const;
};

///
/// \brief CombineOp: CombineOp(OpOperand)
///
class IR_API CombineOp : public pir::Op<CombineOp> {
 public:
  using Op::Op;

  static const char *name() { return "builtin.combine"; }

  static constexpr uint32_t attributes_num = 0;

  static constexpr const char **attributes_name = nullptr;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    const std::vector<Value> &inputs);

  void VerifySig() const;
  std::vector<pir::Value> inputs() {
    std::vector<pir::Value> inputs;
    for (uint32_t idx = 0; idx < num_operands(); idx++) {
      inputs.push_back(operand_source(static_cast<int>(idx)));
    }
    return inputs;
  }
  pir::Value out() { return result(0); }
};

///
/// \brief SliceOp: SliceOp(OpOperand)
///
class IR_API SliceOp : public pir::Op<SliceOp> {
 public:
  using Op::Op;

  static const char *name() { return "builtin.slice"; }

  static constexpr uint32_t attributes_num = 1;

  static const char *attributes_name[attributes_num];

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value input,
                    int index);

  void VerifySig() const;
  pir::Value input() { return operand_source(0); }
  void RefreshStopGradients();

 private:
  static void PassStopGradients(OperationArgument &argument,  // NOLINT
                                int index);
};

///
/// \brief SplitOp: SplitOp(OpOperand)
///
class IR_API SplitOp : public pir::Op<SplitOp> {
 public:
  using Op::Op;

  static const char *name() { return "builtin.split"; }

  static constexpr uint32_t attributes_num = 0;

  static constexpr const char **attributes_name = nullptr;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value input);

  void VerifySig() const;
  pir::Value input() { return operand_source(0); }
  std::vector<Value> outputs() {
    std::vector<Value> res;
    for (uint32_t idx = 0; idx < num_results(); idx++) {
      res.push_back(result(idx));
    }
    return res;
  }
  void RefreshStopGradients();

 private:
  static void PassStopGradients(OperationArgument &argument);  // NOLINT
};

class IR_API ConstantLikeTrait : public OpTraitBase<ConstantLikeTrait> {
 public:
  explicit ConstantLikeTrait(const Operation *op)
      : OpTraitBase<ConstantLikeTrait>(op) {}
};

///
/// \brief ConstantOp
///
class IR_API ConstantOp : public Op<ConstantOp, ConstantLikeTrait> {
 public:
  using Op::Op;
  static const char *name() { return "builtin.constant"; }

  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Attribute value,
                    Type output_type);

  void VerifySig() const;
  Value out() { return result(0); }
  Attribute value() const;
};

///
/// \brief ConstantTensorOp: OpResult = ConstantTensorOp({StrAttribute,
/// StrAttribute})
///
class IR_API ConstantTensorOp : public ConstantOp {
 public:
  using ConstantOp::ConstantOp;

  static ConstantTensorOp dyn_cast(const Operation *op);
  static bool classof(const Operation *op);

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    const std::string &name,
                    Type output_type);

  void VerifySig() const;

  std::string tensor_name();
};

IR_API void PassStopGradientsDefaultly(OperationArgument &argument);  // NOLINT
IR_API void TrueStopGradientsDefaultly(OperationArgument &argument);  // NOLINT
IR_API void RefreshStopGradientsDefaultly(Operation *Op);
}  // namespace pir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::ModuleOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::ParameterOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::SetParameterOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::ShadowOutputOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::CombineOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::SliceOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::SplitOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::ConstantLikeTrait)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::ConstantOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::ConstantTensorOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::GroupOp)
