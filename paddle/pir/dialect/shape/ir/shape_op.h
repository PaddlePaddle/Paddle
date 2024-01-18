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

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::DimOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::TensorDimOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::ShapeOfOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::FromElementsOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::ExtractOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::ConstantIndexOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::IndexCastOp);
