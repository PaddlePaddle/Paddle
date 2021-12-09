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

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace pd {

class PaddleDialect : public Dialect {
 public:
  explicit PaddleDialect(MLIRContext* context);

  static StringRef getDialectNamespace() { return "pd"; }

  /// A hook used to materialize constant values with the given type.
  Operation* materializeConstant(OpBuilder& builder,
                                 Attribute value,
                                 Type type,
                                 Location loc) override;

  Type parseType(DialectAsmParser& parser) const override {
    return Dialect::parseType(parser);
  }
  void printType(Type type, DialectAsmPrinter& printer) const override {
    Dialect::printType(type, printer);
  }
};

}  // namespace pd
}  // namespace mlir
