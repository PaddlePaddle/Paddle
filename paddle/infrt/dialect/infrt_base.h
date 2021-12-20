// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Types.h>

#include "paddle/infrt/dialect/infrt_base.hpp.inc"

namespace infrt::dialect {

class INFRTDialect : public ::mlir::Dialect {
  explicit INFRTDialect(::mlir::MLIRContext *context)
      : ::mlir::Dialect(getDialectNamespace(),
                        context,
                        ::mlir::TypeID::get<INFRTDialect>()) {
    initialize();
  }

  // parse types registered to the dialect.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  // print types registered to the dialect.
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;

  void initialize();
  friend class ::mlir::MLIRContext;

 public:
  static ::llvm::StringRef getDialectNamespace() { return "infrt"; }
};

}  // namespace infrt::dialect

namespace mlir {

template <typename T>
static mlir::IntegerAttr createI32Attr(mlir::OpBuilder &b,  // NOLINT
                                       mlir::Location loc,
                                       T constant) {
  return b.getIntegerAttr(b.getI32Type(), constant);
}

static mlir::ValueRange cvtValueToValueRange(const mlir::Value &operand) {
  return mlir::ValueRange(operand);
}

static mlir::ValueRange concatTwoValueRange(mlir::ValueRange operand_0,
                                            mlir::ValueRange operand_1) {
  mlir::SmallVector<::mlir::Value, 4> operands;
  operands.append(operand_0.begin(), operand_0.end());
  operands.append(operand_1.begin(), operand_1.end());
  return operands;
}

}  // namespace mlir
