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

#include "paddle/infrt/dialect/infrt/ir/basic_kernels.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Support/LogicalResult.h>

#include "paddle/infrt/dialect/dense_tensor.h"

namespace infrt {
namespace dialect {
using namespace mlir;  // NOLINT

static ParseResult parseConstantOp(Type attrType,
                                   OpAsmParser &parser,       // NOLINT
                                   OperationState &result) {  // NOLINT
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(valueAttr, attrType, "value", result.attributes) ||
      parser.addTypeToList(attrType, result.types))
    return failure();
  return success();
}

static ParseResult parseConstantF32Op(OpAsmParser &parser,       // NOLINT
                                      OperationState &result) {  // NOLINT
  return parseConstantOp(
      FloatType::getF32(result.getContext()), parser, result);
}
static ParseResult parseConstantF64Op(OpAsmParser &parser,       // NOLINT
                                      OperationState &result) {  // NOLINT
  return parseConstantOp(
      FloatType::getF64(result.getContext()), parser, result);
}
static ParseResult parseConstantI32Op(OpAsmParser &parser,       // NOLINT
                                      OperationState &result) {  // NOLINT
  return parseConstantOp(
      IntegerType::get(result.getContext(), 32), parser, result);
}
static ParseResult parseConstantI64Op(OpAsmParser &parser,       // NOLINT
                                      OperationState &result) {  // NOLINT
  return parseConstantOp(
      IntegerType::get(result.getContext(), 64), parser, result);
}

static void printConstant(OpAsmPrinter &p, mlir::Operation *op) {  // NOLINT
  p << " ";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});

  if (op->getAttrs().size() > 1) p << ' ';
  Attribute attr = op->getAttr("value");
  if (auto int_attr = attr.dyn_cast<IntegerAttr>()) {
    bool is_signed = int_attr.getType().isIndex() ||
                     int_attr.getType().getIntOrFloatBitWidth() != 1;
    int_attr.getValue().print(p.getStream(), is_signed);
  } else if (auto float_attr = attr.dyn_cast<FloatAttr>()) {
    p << float_attr.getValue().convertToFloat();
  } else {
    op->emitOpError("unknown attribute type");
  }
}

static void print(OpAsmPrinter &p, ConstantF32Op op) {  // NOLINT
  printConstant(p, op);
}
static void print(OpAsmPrinter &p, ConstantF64Op op) {  // NOLINT
  printConstant(p, op);
}
static void print(OpAsmPrinter &p, ConstantI32Op op) {  // NOLINT
  printConstant(p, op);
}
static void print(OpAsmPrinter &p, ConstantI64Op op) {  // NOLINT
  printConstant(p, op);
}

static LogicalResult verify(ConstantF32Op op) { return success(); }
static LogicalResult verify(ConstantI32Op op) { return success(); }
static LogicalResult verify(ConstantF64Op op) { return success(); }
static LogicalResult verify(ConstantI64Op op) { return success(); }

}  // namespace dialect
}  // namespace infrt

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/infrt/ir/basic_kernels.cpp.inc"
