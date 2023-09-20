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
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/op_base.h"

namespace pir {
namespace dialect {

class IR_API SymbolicDim : public Op<SymbolicDim> {
 public:
  using Op::Op;
  static const char *name() { return "shape.SymbolicDim"; }

  static constexpr uint32_t attributes_num = 6;
  static const char *attributes_name[attributes_num];

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    const std::string &sym_name,
                    int64_t value = ShapedTypeInterface::kDynamic,
                    bool knownNonNegative = false,
                    bool knownNegativeOne = false,
                    bool knownNonSizeOne = false,
                    bool knownNonSizeZero = false);
  const std::string getSymName();
  int64_t getValue();
  bool getKnownNonNegative();
  bool getKnownNegativeOne();
  bool getKnownNonSizeOne();
  bool getKnownNonSizeZero();

  void updateSymName(std::string attrValue);
  void updateValue(int64_t attrValue);
  void updateKnownNonNegative(bool attrValue);
  void updateKnownNegativeOne(bool attrValue);
  void updateKnownNonSizeOne(bool attrValue);
  void updateKnownNonSizeZero(bool attrValue);

  bool IsDynamic();
  bool Merge(SymbolicDim other);

  static const std::string getSymbolicDimAttrName() {
    return "kSymbolicDimAttr";
  }

  void Verify() {}
};

class IR_API DimOp : public Op<DimOp> {
 public:
  using Op::Op;
  static const char *name() { return "shape.dim"; }

  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    const std::string &name);

  const std::string getName();
  void setName(std::string attrValue);
  OpResult out() { return result(0); }
  void Verify() {}
};

class IR_API TieProductEqualOp : public Op<TieProductEqualOp> {
 public:
  using Op::Op;
  static const char *name() { return "shape.tie_product_equal"; }

  static constexpr uint32_t attributes_num = 2;
  static const char *attributes_name[attributes_num];

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    int64_t lhs_len,
                    int64_t rhs_len,
                    const std::vector<Value> &inputs);
  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    const std::vector<Value> &lhs,
                    const std::vector<Value> &rhs);
  std::vector<Value> lhs();
  std::vector<Value> rhs();
  void Verify() {}
};

class IR_API TieShapeOp : public Op<TieShapeOp> {
 public:
  using Op::Op;
  static const char *name() { return "shape.tie_shape"; }

  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    pir::Value input);

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value input,
                    const std::vector<Value> &dims);
  Value value();
  std::vector<Value> dims();
  void Verify() {}
};

class IR_API FuncOp : public Op<FuncOp> {
 public:
  using Op::Op;
  static const char *name() { return "shape.func"; }

  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;

  static void Build(Builder &builder,              // NOLINT
                    OperationArgument &argument);  // NOLINT
  void Print(IrPrinter &printer);                  // NOLINT
  Block *block();
  void Verify() {}
};

class IR_API TensorDimOp : public Op<TensorDimOp> {
 public:
  using Op::Op;
  static const char *name() { return "shape.tensor_dim"; }

  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value source,
                    Value index);
  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value source,
                    int64_t index);
  Value index();
  Value source();
  OpResult out() { return result(0); }
  void Verify() {}
};

}  // namespace dialect
}  // namespace pir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::dialect::SymbolicDim);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::dialect::DimOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::dialect::TieProductEqualOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::dialect::TieShapeOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::dialect::FuncOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::dialect::TensorDimOp);
