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

#include <optional>
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/op_base.h"
#include "paddle/pir/core/op_trait.h"

namespace pir::shape {

class IR_API SymbolicDimOp : public Op<SymbolicDimOp> {
 public:
  using Op::Op;
  static const char *name() { return "shape.symbolic_dim"; }

  static constexpr uint32_t attributes_num = 6;
  static const char *attributes_name[attributes_num];

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    const std::string &sym_name,
                    int64_t value = ShapedTypeInterface::kDynamic,
                    bool known_non_negative = false,
                    bool known_negative_one = false,
                    bool known_non_size_one = false,
                    bool known_non_size_zero = false);

  const std::string GetSymName() const;
  int64_t GetDimSize() const;

  bool GetKnownNonNegative();
  bool GetKnownNegativeOne();
  bool GetKnownNonSizeOne();
  bool GetKnownNonSizeZero();

  void SetSymName(const std::string &attr_value);
  void SetDimSize(int64_t attr_value);

  // Sets `known_non_negative` to the value of `flag`
  void UpdateKnownNonNegative(bool flag);

  // Sets `known_negative_one` to the value of `flag`
  void UpdateKnownNegativeOne(bool flag);

  // Sets `known_non_size_one` to the value of `flag`
  void UpdateKnownNonSizeOne(bool flag);

  // Sets `known_non_size_zero` to the value of `flag`
  void UpdateKnownNonSizeZero(bool flag);

  // Returns true if this SymbolicDimOp is not known at compile-time.
  bool IsDynamic() const;

  // Try to merge two SymbolicDimOp.
  bool Merge(SymbolicDimOp other);

  static const std::string GetSymbolicDimAttrName() {
    return "kSymbolicDimAttr";
  }

  void VerifySig() {}
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

  const std::string GetName();
  void SetName(std::string attrValue);
  OpResult out() { return result(0); }
  void VerifySig() {}
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
  void VerifySig() {}
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
  Value input() { return operand_source(0); }
  std::vector<Value> dims();
  void VerifySig() {}
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
  void VerifySig() {}
};

class IR_API TensorDimOp : public Op<TensorDimOp, OneResultTrait> {
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

  Value source() { return operand_source(0); }
  Value index() { return operand_source(1); }
  OpResult out() { return result(0); }
  void VerifySig() {}
  std::optional<int64_t> GetConstantIndex();
};

class IR_API ShapeOfOp : public Op<ShapeOfOp> {
 public:
  using Op::Op;
  static const char *name() { return "shape.shape_of"; }

  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value input);

  Value input() { return operand_source(0); }
  OpResult out() { return result(0); }
  void VerifySig() {}
};

class IR_API FromElementsOp : public Op<FromElementsOp> {
 public:
  using Op::Op;
  static const char *name() { return "shape.from_elements"; }

  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    const std::vector<Value> &elements);

  std::vector<Value> elements();
  OpResult out() { return result(0); }
  void VerifySig() {}
};

class IR_API ExtractOp : public Op<ExtractOp> {
 public:
  using Op::Op;
  static const char *name() { return "shape.extract"; }

  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Value tensor,
                    std::vector<Value> indices);

  Value tensor() { return operand_source(0); }
  std::vector<Value> indices();
  OpResult out() { return result(0); }
  void VerifySig() {}
};

// Specialization of `constant` op that returns an integer of index type.
class IR_API ConstantIndexOp : public ConstantOp {
 public:
  using ConstantOp::ConstantOp;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    int64_t value);
};

// Cast between index and integer types.
class IR_API IndexCastOp : public Op<IndexCastOp> {
 public:
  using Op::Op;
  static const char *name() { return "shape.index_cast"; }

  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;

  static void Build(Builder &builder,             // NOLINT
                    OperationArgument &argument,  // NOLINT
                    Type out,
                    Value in);

  Value in() { return operand_source(0); }
  OpResult out() { return result(0); }
  void VerifySig() {}
};

}  // namespace pir::shape

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::SymbolicDimOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::DimOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::TieProductEqualOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::TieShapeOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::FuncOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::TensorDimOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::ShapeOfOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::FromElementsOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::ExtractOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::ConstantIndexOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::IndexCastOp);
