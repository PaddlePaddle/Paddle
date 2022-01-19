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

#include "paddle/infrt/dialect/dense_tensor.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Support/LogicalResult.h>

#include <tuple>

#include "paddle/infrt/common/global.h"
#include "paddle/infrt/dialect/tensor_shape.h"

namespace infrt {
namespace dt {
void DTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "paddle/infrt/dialect/dense_tensor.cpp.inc"
      >();
}

llvm::Optional<TargetType> GetTargetType(mlir::StringRef key) {
  if (key.equals_insensitive("x86"))
    return TargetType::X86;
  else if (key.equals_insensitive("cuda"))
    return TargetType::CUDA;
  else
    return llvm::None;
}

llvm::Optional<LayoutType> GetLayoutType(mlir::StringRef key) {
  if (key.equals_insensitive("nchw"))
    return LayoutType::NCHW;
  else if (key.equals_insensitive("nhwc"))
    return LayoutType::NHWC;
  else
    return llvm::None;
}

llvm::Optional<PrecisionType> GetPrecisionType(mlir::StringRef key) {
  if (key.equals_insensitive("i32"))
    return PrecisionType::I32;
  else if (key.equals_insensitive("f32"))
    return PrecisionType::F32;
  else
    return llvm::None;
}

TensorType TensorType::get(TargetType target,
                           LayoutType layout,
                           PrecisionType precision) {
  return Base::get(
      ::infrt::Global::getMLIRContext(), target, layout, precision);
}

TargetType TensorType::target() { return getImpl()->target_; }

LayoutType TensorType::layout() { return getImpl()->layout_; }

PrecisionType TensorType::precision() { return getImpl()->precision_; }

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, TensorType tensorType) {
  os << "TensorType<" << tensorType.target() << ", " << tensorType.layout()
     << ", " << tensorType.precision() << ">";
  return os;
}

TensorMapType TensorMapType::get() {
  return Base::get(::infrt::Global::getMLIRContext());
}

TensorMapType TensorMapType::get(mlir::MLIRContext *context) {
  return Base::get(context);
}

StringType StringType::get() {
  return Base::get(::infrt::Global::getMLIRContext());
}

StringType StringType::get(mlir::MLIRContext *context) {
  return Base::get(context);
}

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, TargetType type) {
  switch (type) {
    case (TargetType::X86):
      os << "X86";
      break;
    case (TargetType::CUDA):
      os << "CUDA";
      break;
    default:
      os << "Unsupported";
  }
  return os;
}

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, LayoutType type) {
  switch (type) {
    case (LayoutType::NCHW):
      os << "NCHW";
      break;
    case (LayoutType::NHWC):
      os << "NHWC";
      break;
    default:
      os << "Unsupported";
  }
  return os;
}

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, PrecisionType type) {
  switch (type) {
    case (PrecisionType::I32):
      os << "I32";
      break;
    case (PrecisionType::F32):
      os << "F32";
      break;
    default:
      os << "Unsupported";
  }
  return os;
}

static mlir::Type getTensorType(mlir::MLIRContext *context) {
  auto t_dialect = mlir::Identifier::get("t", context);
  return mlir::OpaqueType::get(t_dialect, "tensor");
}

static mlir::ParseResult parseCreateUninitTensorOp(
    mlir::OpAsmParser &parser,       // NOLINT
    mlir::OperationState &result) {  // NOLINT
  auto loc = parser.getCurrentLocation();
  mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<mlir::Type> outputTypes(outputRawTypes);

  mlir::ArrayAttr shapeAttr;
  if (parser.parseAttribute(shapeAttr,
                            parser.getBuilder().getI64Type(),
                            "shape",
                            result.attributes))
    return mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes)) return mlir::failure();

  if (parser.parseArrow()) return mlir::failure();
  if (parser.parseType(outputRawTypes[0])) return mlir::failure();
  if (!outputRawTypes[0].isa<TensorType>())
    return parser.emitError(loc, "invalid kind of type specified");
  result.addTypes(outputTypes);
  return mlir::success();
}

template <typename CreateUninitTensorOp>
static void printCreateUninitTensorOp(mlir::OpAsmPrinter &p,  // NOLINT
                                      CreateUninitTensorOp op) {
  p << CreateUninitTensorOp::getOperationName();
  p << " ";
  p.printAttributeWithoutType(op.shapeAttr());
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"shape"});
  p << " -> ";
  p << op.getOperation()->getResultTypes();
}

static mlir::ParseResult parseSetTensorOp(
    mlir::OpAsmParser &parser,       // NOLINT
    mlir::OperationState &result) {  // NOLINT
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 1> operands;
  if (parser.parseOperandList(operands, 1)) return mlir::failure();

  auto tensor_type = getTensorType(result.getContext());

  mlir::Attribute value_attr;
  return mlir::failure(
      parser.resolveOperand(operands[0], tensor_type, result.operands) ||
      parser.parseAttribute(value_attr, "values", result.attributes));
}

template <typename SetTensorOp>
static void printSetTensorOp(mlir::OpAsmPrinter &p, SetTensorOp op) {  // NOLINT
  p << SetTensorOp::getOperationName() << " ";
  p.printOperand(op.getOperand());
  p << " " << op->getAttr("values");
}
}  // namespace dt
}  // namespace infrt

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/dense_tensor.cpp.inc"  // NOLINT

#include "paddle/infrt/dialect/dense_tensor_dialect.cpp.inc"
