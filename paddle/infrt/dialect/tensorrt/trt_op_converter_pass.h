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
#include <mlir/Pass/Pass.h>

namespace infrt {
namespace trt {
/*
 * trtOpConverterPass.
 *
 * source ir:
 * func @main() -> tensor<?xf32> {
 *   %a = "pd.feed"()...
 *   %d, %f = "pd.graph"(%a) {
 *     %m = "pd.conv2d"(%a)...
 *     %n = "pd.conv3d"(%m)...
 *     %s = "pd.conv2d"(%a)...
 *     "pd.return" %n, %s
 *   } ...
 *   "pd.fetch" %d, %f
 * }
 *
 * destination ir:
 * func @main() -> tensor<?xf32> {
 *   %a = "pd.feed"()...
 *   %d, %f = "pd.graph"(%a) {
 *     %m = "trt.Convolution"(%a)...
 *     %n = "trt.Convolution"(%m)...
 *     %s = "trt.Convolution"(%a)...
 *     "trt.return" %n, %s
 *   } ...
 *   "pd.fetch" %d, %f
 * }
 */
class trtOpConverterPass
    : public mlir::PassWrapper<trtOpConverterPass,
                               mlir::OperationPass<mlir::FuncOp>> {
 public:
  ::llvm::StringRef getName() const override { return "trtOpTellerPass"; }
  void runOnOperation() final;
};
}  // namespace trt
}  // namespace infrt
