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

#include "paddle/ap/include/axpr/attr_map.h"
#include "paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_symbolic_shape.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace ap::dialect {

class IR_API FacadeOp
    : public pir::Op<FacadeOp, ::paddle::dialect::InferSymbolicShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "ap_op.facade"; }
  static constexpr uint32_t attributes_num = 3;
  static const char *attributes_name[attributes_num];
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    const std::vector<pir::Value> &inputs,
                    const pir::AttributeMap &attributes,
                    const std::vector<pir::Type> &output_types);
  void VerifySig() const {}
  bool InferSymbolicShape(pir::InferSymbolicShapeContext *infer_context);
};

class IR_API UpSpiderOp
    : public pir::Op<UpSpiderOp,
                     pir::SideEffectTrait,
                     pir::ImmutableLayoutTrait,
                     ::paddle::dialect::InferSymbolicShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "ap_op.up_spider"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value lhs,
                    pir::Value rhs);
  void VerifySig() const {}
  bool InferSymbolicShape(pir::InferSymbolicShapeContext *infer_context) {
    return true;
  }
};

class IR_API DownSpiderOp
    : public pir::Op<DownSpiderOp,
                     ::paddle::dialect::InferSymbolicShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "ap_op.down_spider"; }
  static constexpr uint32_t attributes_num = 0;
  static constexpr const char **attributes_name = nullptr;
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value input);
  void VerifySig() const {}
  bool InferSymbolicShape(pir::InferSymbolicShapeContext *infer_context);
};

class IR_API LoadFromRegisterOp
    : public pir::Op<LoadFromRegisterOp,
                     ::paddle::dialect::InferSymbolicShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "ap_op.load_from_register"; }
  static constexpr uint32_t attributes_num = 4;
  static const char *attributes_name[attributes_num];
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Type output_type,
                    const symbol::ShapeOrDataDimExprs &shape_or_data,
                    const std::string &name,
                    const std::string &register_var_name);
  void VerifySig() const {}
  bool InferSymbolicShape(pir::InferSymbolicShapeContext *infer_context);
};

class IR_API StoreToRegisterOp
    : public pir::Op<StoreToRegisterOp,
                     pir::SideEffectTrait,
                     pir::ImmutableLayoutTrait,
                     ::paddle::dialect::InferSymbolicShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "ap_op.store_to_register"; }
  static constexpr uint32_t attributes_num = 2;
  static const char *attributes_name[attributes_num];
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value input,
                    const std::string &name,
                    const std::string &register_var_name);
  void VerifySig() const {}
  bool InferSymbolicShape(pir::InferSymbolicShapeContext *infer_context);
};

class IR_API LoadFromGlobalOp
    : public pir::Op<LoadFromGlobalOp,
                     ::paddle::dialect::InferSymbolicShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "ap_op.load_from_global"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value input,
                    const std::string &index_func_unique_id);
  void VerifySig() const {}
  bool InferSymbolicShape(pir::InferSymbolicShapeContext *infer_context);
};

class IR_API StoreToGlobalOp
    : public pir::Op<StoreToGlobalOp,
                     pir::SideEffectTrait,
                     pir::ImmutableLayoutTrait,
                     ::paddle::dialect::InferSymbolicShapeInterface> {
 public:
  using Op::Op;
  static const char *name() { return "ap_op.store_to_global"; }
  static constexpr uint32_t attributes_num = 1;
  static const char *attributes_name[attributes_num];
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value var,
                    pir::Value val,
                    const std::string &index_func_unique_id);
  void VerifySig() const {}
  bool InferSymbolicShape(pir::InferSymbolicShapeContext *infer_context);
};

}  // namespace ap::dialect

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ap::dialect::FacadeOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ap::dialect::UpSpiderOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ap::dialect::DownSpiderOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ap::dialect::LoadFromRegisterOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ap::dialect::StoreToRegisterOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ap::dialect::LoadFromGlobalOp);
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ap::dialect::StoreToGlobalOp);
