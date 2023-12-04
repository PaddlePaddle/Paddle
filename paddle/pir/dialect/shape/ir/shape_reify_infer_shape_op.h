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

#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/core/infer_type_op_interface.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/op_base.h"
#include "paddle/pir/core/op_trait.h"
#include "paddle/pir/dialect/shape/ir/shape_reify_infer_shape_op.h"

namespace pir::shape {

class IR_API AbsOp : public Op<AbsOp, InferShapedTypeOpInterface> {
 public:
  using Op::Op;
  static const char *name() { return "shape.abs"; }

  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value x);

  void VerifySig() {}
  Value x() { return operand_source(0); }
  OpResult out() { return result(0); }
  bool ReifyReturnTypeShapes(
      Builder &builder,  // NOLINT
      const std::vector<OpOperand> &operands,
      std::vector<Value> &reified_return_shapes);  // NOLINT
};

class IR_API TransposeOp : public Op<TransposeOp, InferShapedTypeOpInterface> {
 public:
  using Op::Op;
  static const char *name() { return "shape.transpose"; }

  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value x,
                    std::vector<int> &perm);  // NOLINT

  void VerifySig() {}
  Value x() { return operand_source(0); }
  OpResult out() { return result(0); }
  std::vector<int64_t> permutation();

  bool ReifyReturnTypeShapes(
      Builder &builder,  // NOLINT
      const std::vector<OpOperand> &operands,
      std::vector<Value> &reified_return_shapes);  // NOLINT
};

class IR_API ConcatOp : public Op<ConcatOp, InferShapedTypeOpInterface> {
 public:
  using Op::Op;
  static const char *name() { return "shape.concat"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value x,
                    Value axis = 0);

  void VerifySig() {}
  Value x() { return operand_source(0); }
  Value axis() { return operand_source(1); }
  OpResult out() { return result(0); }
  // TODO(zhangbopd):
  int dimension() { return 0; }

  bool ReifyReturnTypeShapes(
      Builder &builder,  // NOLINT
      const std::vector<OpOperand> &operands,
      std::vector<Value> &reified_return_shapes);  // NOLINT
};

}  // namespace pir::shape

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::AbsOp)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::TransposeOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::ConcatOp);
