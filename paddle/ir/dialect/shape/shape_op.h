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
namespace dialect {

class IR_API SymbolicDim : public Op<SymbolicDim> {
 public:
  using Op::Op;
  static const char *name() { return "shape.SymbolicDim"; }

  static constexpr uint32_t attributes_num = 6;
  static const char *attributes_name[attributes_num];

  static void Build(
      Builder &builder,             // NOLINT
      OperationArgument &argument,  // NOLINT
      const std::string &sym_name,
      int64_t value = -100000,  // TODO(zhangbo): value = ShapedType::kDynamic
      bool knownNonNegative = false,
      bool knownNegativeOne = false,
      bool knownNonSizeOne = false,
      bool knownNonSizeZero = false);
  std::string getSymName();
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
  void Verify() {}
};

}  // namespace dialect
}  // namespace ir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(ir::dialect::SymbolicDim);
