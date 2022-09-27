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
#include <mlir/IR/TypeUtilities.h>
#include <mlir/IR/Types.h>

namespace infrt {

mlir::SmallVector<mlir::Value, 4> cvtValueToValueRange(
    const mlir::Value &operand);

mlir::SmallVector<mlir::Value, 4> concatTwoValueRange(
    mlir::ValueRange operand_0, mlir::ValueRange operand_1);
}  // namespace infrt
