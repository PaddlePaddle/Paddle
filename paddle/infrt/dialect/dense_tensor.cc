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
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Support/LogicalResult.h>

#include <tuple>

#include "paddle/infrt/common/global.h"
#include "paddle/infrt/dialect/tensor_shape.h"

namespace infrt::dt {

void DTDialect::initialize() {
  allowUnknownTypes();
  addOperations<
#define GET_OP_LIST
#include "paddle/infrt/dialect/dense_tensor.cpp.inc"
      >();
}

namespace detail {
struct TensorTypeStorage : public mlir::TypeStorage {
  TensorTypeStorage(TargetType target,
                    LayoutType layout,
                    PrecisionType precision)
      : target_(target), layout_(layout), precision_(precision) {}

  using KeyTy = std::tuple<TargetType, LayoutType, PrecisionType>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(target_, layout_, precision_);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static TensorTypeStorage *construct(
      mlir::TypeStorageAllocator &allocator,  // NOLINT
      const KeyTy &key) {
    return new (allocator.allocate<TensorTypeStorage>())
        TensorTypeStorage(std::get<0>(key), std::get<1>(key), std::get<2>(key));
  }

  TargetType target_;
  LayoutType layout_;
  PrecisionType precision_;
};
}  // namespace detail

llvm::Optional<TargetType> GetTargetType(mlir::StringRef key) {
  if (key.equals_lower("x86"))
    return TargetType::X86;
  else if (key.equals_lower("cuda"))
    return TargetType::CUDA;
  else
    return llvm::None;
}

llvm::Optional<LayoutType> GetLayoutType(mlir::StringRef key) {
  if (key.equals_lower("nchw"))
    return LayoutType::NCHW;
  else if (key.equals_lower("nhwc"))
    return LayoutType::NHWC;
  else
    return llvm::None;
}

llvm::Optional<PrecisionType> GetPrecisionType(mlir::StringRef key) {
  if (key.equals_lower("i32"))
    return PrecisionType::I32;
  else if (key.equals_lower("f32"))
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

raw_ostream &operator<<(raw_ostream &os, TensorType tensorType) {
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

raw_ostream &operator<<(raw_ostream &os, TargetType type) {
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

raw_ostream &operator<<(raw_ostream &os, LayoutType type) {
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

raw_ostream &operator<<(raw_ostream &os, PrecisionType type) {
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

static Type getTensorType(mlir::MLIRContext *context) {
  auto t_dialect = Identifier::get("t", context);
  return OpaqueType::get(t_dialect, "tensor", context);
}

static ParseResult parseCreateUninitTensorOp(
    OpAsmParser &parser,       // NOLINT
    OperationState &result) {  // NOLINT
  auto loc = parser.getCurrentLocation();
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);

  mlir::ArrayAttr shapeAttr;
  if (parser.parseAttribute(shapeAttr,
                            parser.getBuilder().getI64Type(),
                            "shape",
                            result.attributes))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  if (parser.parseArrow()) return failure();
  if (parser.parseType(outputRawTypes[0])) return failure();
  if (!outputRawTypes[0].isa<TensorType>())
    return parser.emitError(loc, "invalid kind of type specified");
  result.addTypes(outputTypes);
  return success();
}

template <typename CreateUninitTensorOp>
static void printCreateUninitTensorOp(OpAsmPrinter &p,  // NOLINT
                                      CreateUninitTensorOp op) {
  p << CreateUninitTensorOp::getOperationName();
  p << " ";
  p.printAttributeWithoutType(op.shapeAttr());
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"shape"});
  p << " -> ";
  p << op.getOperation()->getResultTypes();
}

// TODO(shibo): can be removed?
// static ParseResult parseFillTensorWithConstantOp(OpAsmParser& parser,
// OperationState& result) {
//  auto loc = parser.getCurrentLocation();
//  ::mlir::OpAsmParser::OperandType inputRawOperands[1];
//  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType>
//  inputOperands(inputRawOperands);
//  ::mlir::Type inputRawTypes[1];
//  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);
//
//  if (parser.parseOperand(inputRawOperands[0])) return failure();
//
//  if (parser.parseColon()) return failure();
//  if (parser.parseType(inputRawTypes[0])) return failure();
//  if (!inputRawTypes[0].isa<TensorType>())
//    return parser.emitError(loc, "invalid kind of type specified");
//
//  Attribute value_attr;
//  if (parser.resolveOperands(inputOperands, inputTypes, loc, result.operands))
//  return failure();
//  if (parser.parseAttribute(value_attr, "value", result.attributes)) return
//  failure();
//  return success();
//}

// TODO(shibo): can be removed?
// template <typename FillTensorOp>
// static void printFillTensorWithConstantOp(OpAsmPrinter& p, FillTensorOp op) {
//  p << FillTensorOp::getOperationName();
//  p << " ";
//  p.printOperand(op.getOperand());
//  p << " : ";
//  p << op.getOperation()->getOperandTypes();
//  p << " ";
//  p << op.getAttr("value");
//}

static ParseResult parseSetTensorOp(OpAsmParser &parser,       // NOLINT
                                    OperationState &result) {  // NOLINT
  SmallVector<OpAsmParser::OperandType, 1> operands;
  if (parser.parseOperandList(operands, 1)) return failure();

  auto tensor_type = getTensorType(result.getContext());

  Attribute value_attr;
  return failure(
      parser.resolveOperand(operands[0], tensor_type, result.operands) ||
      parser.parseAttribute(value_attr, "values", result.attributes));
}

template <typename SetTensorOp>
static void printSetTensorOp(OpAsmPrinter &p, SetTensorOp op) {  // NOLINT
  p << SetTensorOp::getOperationName() << " ";
  p.printOperand(op.getOperand());
  p << " " << op.getAttr("values");
}

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/dense_tensor.cpp.inc"  // NOLINT

}  // namespace infrt::dt
