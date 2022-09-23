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
#include <mlir/Pass/Pass.h>

namespace infrt {
namespace trt {

std::unique_ptr<mlir::Pass> CreateTrtGraphFusePass();

/*
 * trtGraphFusePass.
 *
 * Merge the adjacent graph op to a new graph op.
 *
 * source func:
 *
 * func @main(%a : tensor<?xf32>) -> tensor<?xf32> {
 *  %c = "infrt.graph"(%a) {
 *     %m = "pd.conv2d"(%a)...
 *     infrt.return %m...
 *  } ...
 *  %d = "infrt.graph"(%c) {
 *      %m = "pd.conv3d"(%c)...
 *      infrt.return %m...
 *  } ...
 *  %f = "infrt.graph"(%a) {
 *      %m = "pd.conv2d"(%a)...
 *      infrt.return %m...
 *  } ...
 *  infrt.return %d, %f :...
 * }
 *
 * destination func:
 * func @main(%a : tensor<?xf32>) -> tensor<?xf32> {
 *  %d, %f = "infrt.graph"(%a) {
 *     %m = "pd.conv2d"(%a)...
 *     %n = "pd.conv3d"(%m)...
 *     %s = "pd.conv2d"(%a)...
 *     infrt.return %n, %s:...
 *  } ...
 *  infrt.return %d, %f:...
 * }
 */
class TRTGraphFusePass
    : public mlir::PassWrapper<TRTGraphFusePass, mlir::FunctionPass> {
 public:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {}
  ::llvm::StringRef getName() const override { return "trtGraphFusePass"; }
  void runOnFunction() override;
};
}  // namespace trt
}  // namespace infrt
