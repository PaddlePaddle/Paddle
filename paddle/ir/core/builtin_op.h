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

#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/op_base.h"

namespace ir {

class Program;
class Block;
///
/// \brief ModuleOp
///
class IR_API ModuleOp : public ir::Op<ModuleOp> {
 public:
  using Op::Op;
  static const char *name() { return "builtin.module"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  void Verify() const;
  Program *program();
  Block *block();

  //
  // As the top operation, ModuleOp only support create&destroye through
  // below interface: "create"&"destroy".
  static ModuleOp Create(IrContext *context, Program *pointer);
  void Destroy();
};

///
/// \brief GetParameterOp: OpResult = GetParameterOp({StrAttribute,
/// StrAttribute})
///
class IR_API GetParameterOp : public ir::Op<GetParameterOp> {
 public:
  using Op::Op;
  static const char *name() { return "builtin.get_parameter"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    const std::string &name,
                    Type type);
  void Verify() const;
};

///
/// \brief SetParameterOp: SetParameterOp(OpOperand, {StrAttribute,
/// StrAttribute})
///
class IR_API SetParameterOp : public ir::Op<SetParameterOp> {
 public:
  using Op::Op;
  static const char *name() { return "builtin.set_parameter"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    OpResult parameter,
                    const std::string &name);
  void Verify() const;
};

///
/// \brief CombineOp: CombineOp(OpOperand)
///
class IR_API CombineOp : public ir::Op<CombineOp> {
 public:
  using Op::Op;

  static const char *name() { return "builtin.combine"; }

  static constexpr uint32_t attributes_num = 0;

  static constexpr const char **attributes_name = nullptr;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    const std::vector<ir::OpResult> &inputs);

  void Verify() const;
  ir::OpResult out() { return result(0); }
};

///
/// \brief SliceOp: SliceOp(OpOperand)
///
class IR_API SliceOp : public ir::Op<SliceOp> {
 public:
  using Op::Op;

  static const char *name() { return "builtin.slice"; }

  static constexpr uint32_t attributes_num = 1;

  static const char *attributes_name[attributes_num];
  void Verify() const;
  ir::OpResult out() { return result(0); }
};

class IR_API ConstantLikeTrait : public OpTraitBase<ConstantLikeTrait> {
 public:
  explicit ConstantLikeTrait(Operation *op)
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

  void Verify() const;

  Attribute value() const;
};

}  // namespace ir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::ModuleOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::GetParameterOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::SetParameterOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::CombineOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::SliceOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::ConstantLikeTrait)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::ConstantOp)
