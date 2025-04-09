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
#include <typeindex>
#include <typeinfo>

#include "paddle/common/macros.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/utils/test_macros.h"

namespace paddle {
namespace dialect {

class TEST_API OperatorDialect : public pir::Dialect {
 public:
  explicit OperatorDialect(pir::IrContext* context);

  static const char* name() { return "pd_op"; }

  void PrintType(pir::Type type, std::ostream& os) const override;
  void PrintAttribute(pir::Attribute attr, std::ostream& os) const override;

  pir::OpPrintFn PrintOperation(
      const pir::Operation& op) const override;  // NOLINT

 private:
  void initialize();
};

inline bool IsCustomOp(pir::Operation* op) {
  std::string op_name = op->name();
  return op_name.find("custom_op") != op_name.npos;
}

inline bool IsCustomEngineOp(pir::Operation* op) {
  std::string op_name = op->name();
  return op_name.find("custom_engine") != op_name.npos;
}

inline bool IsInplaceOp(pir::Operation* op) {
  std::string op_name = op->name();
  return op_name.back() == '_';
}

inline bool IsTensorRTOp(pir::Operation* op) {
  std::string op_name = op->name();
  return op_name == "pd_op.tensorrt_engine";
}

class CustomOpDialect : public pir::Dialect {
 public:
  explicit CustomOpDialect(pir::IrContext* context);

  static const char* name() { return "custom_op"; }

  void PrintType(pir::Type type, std::ostream& os) const override;
  void PrintAttribute(pir::Attribute type, std::ostream& os) const override;

  pir::OpPrintFn PrintOperation(
      const pir::Operation& op) const override;  // NOLINT

  void RegisterCustomOp(const paddle::OpMetaInfo& op_meta);

  bool HasRegistered(const std::string& op_name) {
    if (std::find(op_names_.begin(), op_names_.end(), op_name) !=
        op_names_.end()) {
      return true;
    }
    return false;
  }

 private:
  std::vector<const char*> op_names_;
};

class TEST_API CustomEngineDialect : public pir::Dialect {
 public:
  explicit CustomEngineDialect(pir::IrContext* context);

  static const char* name() { return "custom_engine"; }

  void PrintType(pir::Type type, std::ostream& os) const override;
  void PrintAttribute(pir::Attribute type, std::ostream& os) const override;

  pir::OpPrintFn PrintOperation(
      const pir::Operation& op) const override;  // NOLINT

  // template <typename ConcreteOp>
  // static void RegisterCustomEngineOp();

  bool HasRegistered(const std::string& op_name) {
    if (std::find(op_names_.begin(), op_names_.end(), op_name) !=
        op_names_.end()) {
      return true;
    }
    return false;
  }

 private:
  std::vector<const char*> op_names_;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::OperatorDialect)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::CustomOpDialect)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::CustomEngineDialect)
