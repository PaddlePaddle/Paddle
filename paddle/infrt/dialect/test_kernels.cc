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

#include "paddle/infrt/dialect/test_kernels.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/TypeUtilities.h>

namespace infrt {
namespace dialect {
//===----------------------------------------------------------------------===//
// BenchmarkOp
//===----------------------------------------------------------------------===//

// Parse the BenchmarkOp in the following format
// infrt.benchmark "add.i32"(%c : i32, %d : f32)
//       max_count = 100, duration_secs = 1 {
// ...
// }

static mlir::ParseResult parseBenchmarkOp(
    mlir::OpAsmParser &parser,       // NOLINT
    mlir::OperationState &result) {  // NOLINT
  mlir::StringAttr nameAttr;
  if (parser.parseAttribute(nameAttr, "name", result.attributes))
    return mlir::failure();

  // Parse the operands, e.g. (%c : i32, %d : f32)
  if (parser.parseLParen()) return mlir::failure();

  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> operands;
  llvm::SmallVector<mlir::Type, 4> types;
  llvm::SMLoc type_loc = parser.getCurrentLocation();

  if (parser.parseOptionalRParen()) {
    // Parse non-empty operands
    do {
      // Parse %c : i32,
      mlir::OpAsmParser::OperandType operand;
      mlir::Type type;

      if (parser.parseOperand(operand) || parser.parseColonType(type))
        return mlir::failure();

      operands.push_back(operand);
      types.push_back(type);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen()) return mlir::failure();
  }

  if (parser.resolveOperands(operands, types, type_loc, result.operands))
    return mlir::failure();

  // Parse the keyword attribute, e.g. max_count = 100, duration_secs = 1
  do {
    mlir::StringRef attr;
    mlir::Attribute resultAttr;
    if (parser.parseKeyword(&attr) || parser.parseEqual() ||
        parser.parseAttribute(resultAttr,
                              parser.getBuilder().getIntegerType(32),
                              attr,
                              result.attributes))
      return mlir::failure();
  } while (mlir::succeeded(parser.parseOptionalComma()));

  // Set the default attribute num_warmup_runs to 1 if unset
  auto setDefaultAttrIfUnset = [&](const char *attr_name, int value) {
    bool found = llvm::any_of(result.attributes,
                              [attr_name](const mlir::NamedAttribute &attr) {
                                return attr.getName() == attr_name;
                              });
    if (!found) {
      mlir::IntegerAttr default_val =
          parser.getBuilder().getI32IntegerAttr(value);
      result.addAttribute(attr_name, default_val);
    }
  };
  setDefaultAttrIfUnset("num_warmup_runs", 1);

  mlir::Region *target = result.addRegion();
  return parser.parseRegion(*target,
                            operands,
                            types,
                            /*enableNameShadowing=*/true);
}

// Print the BenchmarkOp in the following format
// infrt.benchmark "add.i32"(%c : i32, %d : f32)
//       max_count = 100, duration_secs = 1 {
// ...
// }
static void print(mlir::OpAsmPrinter &p, BenchmarkOp op) {  // NOLINT
  p << "infrt.benchmark ";

  // Print the name attribute, e.g "add.i32"
  auto name_attr = op->getAttr("name");
  p << name_attr;

  // Print the operands and types, e.g. (%c : i32, %d : f32)
  p << '(';
  llvm::interleaveComma(llvm::zip(op.getOperands(), op.getOperandTypes()),
                        p,
                        [&](const auto &it) {
                          p << std::get<0>(it) << " : " << std::get<1>(it);
                        });
  p << ") ";

  bool need_comma = false;
  // Print the attributes, e.g. max_count = 100, duration_secs = 1
  for (auto &name_attr : op->getAttrs()) {
    auto id = name_attr.getName();
    if (id == "name") continue;
    if (need_comma) p << ", ";
    auto attr = name_attr.getValue();
    p << id << " = ";
    if (auto int_attr = attr.dyn_cast<mlir::IntegerAttr>()) {
      int_attr.getValue().print(p.getStream(), /*isSigned=*/false);
    } else {
      op.emitOpError("Unexpected attribute");
    }
    need_comma = true;
  }
  p << ' ';

  // Print the region
  // Reuse the argument names provided to the op for the bbarg names within
  // the region.
  p.shadowRegionArgs(op.region(), op.getOperands());
  p.printRegion(op.region(), /*printEntryBlockArgs=*/false);
}

static mlir::LogicalResult verify(BenchmarkOp op) {
  // Verify that the target benchmark region has exactly one return value.
  auto &region = op.region();
  auto &last_op = region.front().back();
  if (last_op.getName().getStringRef() != "infrt.return") {
    return op.emitOpError("missing return statement");
  }
  if (last_op.getNumOperands() != 1) {
    return op.emitOpError(
        "incorrect number of return values. One return value is expected");
  }

  return mlir::success();
}
}  // namespace dialect
}  // namespace infrt

#define GET_OP_CLASSES
#include "paddle/infrt/dialect/test_kernels.cpp.inc"
