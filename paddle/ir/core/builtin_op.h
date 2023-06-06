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
class ModuleOp : public ir::Op<ModuleOp> {
 public:
  using Op::Op;
  static const char *name() { return "builtin.module"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  static void Verify(const std::vector<ir::OpResult> &inputs,
                     const std::vector<ir::Type> &outputs,
                     const ir::AttributeMap &attributes);

  Program *program();
  Block *block();

  //
  // As the top operation, ModuleOp only support create&destroye through
  // below interface: "create"&"destroy".
  static ModuleOp create(IrContext *context, Program *pointer);
  void destroy();
};

///
/// \brief GetParameterOp: OpResult = GetParameterOp({StrAttribute,
/// StrAttribute})
///
class GetParameterOp : public ir::Op<GetParameterOp> {
 public:
  using Op::Op;
  static const char *name() { return "builtin.get_parameter"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  static void Verify(const std::vector<ir::OpResult> &inputs,
                     const std::vector<ir::Type> &outputs,
                     const ir::AttributeMap &attributes);
};

///
/// \brief SetParameterOp: SetParameterOp(OpOperand, {StrAttribute,
/// StrAttribute})
///
class SetParameterOp : public ir::Op<SetParameterOp> {
 public:
  using Op::Op;
  static const char *name() { return "builtin.set_parameter"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  static void Verify(const std::vector<ir::OpResult> &inputs,
                     const std::vector<ir::Type> &outputs,
                     const ir::AttributeMap &attributes);
};

///
/// \brief CombineOp: CombineOp(OpOperand)
///
class CombineOp : public ir::Op<CombineOp> {
 public:
  using Op::Op;

  static const char *name() { return "builtin.combine"; }

  static constexpr uint32_t attributes_num = 0;

  static constexpr const char **attributes_name = nullptr;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    const std::vector<ir::OpResult> &inputs);

  static void Verify(const std::vector<ir::OpResult> &inputs,
                     const std::vector<ir::Type> &outputs,
                     const ir::AttributeMap &attributes);
};

///
/// \brief SliceOp: SliceOp(OpOperand)
///
class SliceOp : public ir::Op<SliceOp> {
 public:
  using Op::Op;

  static const char *name() { return "builtin.slice"; }

  static constexpr uint32_t attributes_num = 1;

  static const char *attributes_name[attributes_num];
  static void Verify(const std::vector<ir::OpResult> &inputs,
                     const std::vector<ir::Type> &outputs,
                     const ir::AttributeMap &attributes);
};

class ConstantLikeTrait : public OpTraitBase<ConstantLikeTrait> {
 public:
  explicit ConstantLikeTrait(Operation *op)
      : OpTraitBase<ConstantLikeTrait>(op) {}
};

///
/// \brief ConstantOp
///
class ConstantOp : public Op<ConstantOp, ConstantLikeTrait> {
 public:
  using Op::Op;
  static const char *name() { return "builtin.constant"; }

  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Attribute value,
                    Type output_type);

  static void Verify(const std::vector<ir::OpResult> &inputs,
                     const std::vector<ir::Type> &outputs,
                     const AttributeMap &attributes);

  Attribute value();
};

}  // namespace ir
