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

#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "paddle/infrt/dialect/core/ir/core_dialect.h"
#include "paddle/infrt/dialect/tensorrt/trt_ops.h"

namespace infrt {
namespace trt {
/*
 * trtOpConverterPass.
 *
 * source ir:
 * func @main(%a : tensor<?xf32>) -> tensor<?xf32> {
 *   %d, %f = "infrt.graph"(%a) {
 *     %m = "pd.conv2d"(%a)...
 *     %n = "pd.conv3d"(%m)...
 *     %s = "pd.conv2d"(%a)...
 *     core.return %n, %s:...
 *   } ...
 *   core.return %d, %f:...
 * }
 *
 * destination ir:
 * func @main(%a : tensor<?xf32>) -> tensor<?xf32> {
 *   %engine = "trt.create_engine"(%a) ({
 *     %m = "trt.Convolution"(%a)...
 *     %n = "trt.Convolution"(%m)...
 *     %s = "trt.Convolution"(%a)...
 *     core.return %n, %s :...
 *   }){run_once = true} ...
 *   %d, %f = "trt.execute"(%engine, %a)...
 *   core.return %d, %f :...
 * }
 */
struct TRTOpConverterPass
    : public mlir::PassWrapper<TRTOpConverterPass,
                               mlir::OperationPass<mlir::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<TensorRTDialect>();
  }
  ::llvm::StringRef getName() const override { return "trtOpConverterPass"; }
  void runOnOperation() final;
};
}  // namespace trt
}  // namespace infrt
