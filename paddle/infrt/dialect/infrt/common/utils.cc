// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/dialect/infrt/common/utils.h"

mlir::SmallVector<mlir::Value, 4> infrt::cvtValueToValueRange(
    const mlir::Value &operand) {
  return mlir::SmallVector<mlir::Value, 4>(1, operand);
}

mlir::SmallVector<mlir::Value, 4> infrt::concatTwoValueRange(
    mlir::ValueRange operand_0, mlir::ValueRange operand_1) {
  mlir::SmallVector<mlir::Value, 4> operands;
  operands.append(operand_0.begin(), operand_0.end());
  operands.append(operand_1.begin(), operand_1.end());
  return operands;
}
