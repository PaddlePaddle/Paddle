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
#include "paddle/infrt/dialect/infrt_base.h"
#include "paddle/infrt/dialect/tensorrt/trt_ops.h"

namespace infrt {
namespace trt {
/*
 * trtOpTellerPass.
 *
 * Pick out the operators supported by tensorrt and convert it to graph.
 *
 * source func:
 *
 * func @main() -> tensor<?xf32> {
 *  %a = "pd.feed"()...
 *  %c = "pd.conv2d"(%a) ...
 *  %d = "pd.conv3d"(%c) ...
 *  %f = "pd.conv2d"(%a) ...
 *  "pd.fetch" (%d, %f)
 * }
 *
 * destination func:
 * func @main() -> tensor<?xf32> {
 *  %a = "pd.feed"()...
 *  %c = "trt.create_engine"(%a) {
 *     %m = "pd.conv2d"(%a)...
 *     "Infrt.return" (%m)
 *  } ...
 *  %d = "trt.create_engine"(%c) {
 *      %m = "pd.conv3d"(%c)...
 *      "Infrt.return" (%m)
 *  } ...
 *  %f = "trt.create_engine"(%a) {
 *      %m = "pd.conv2d"(%a)...
 *      "Infrt.return" (%m)
 *  } ...
 *  "pd.fetch" (%d, %f)
 * }
 * TODO(winter-wang): Supplementary how to judge the operators can be supported
 * by tensorrt.
 */
class TRTOpTellerPass
    : public mlir::PassWrapper<TRTOpTellerPass, mlir::FunctionPass> {
 public:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<TensorRTDialect, ::infrt::dialect::INFRTDialect>();
  }
  ::llvm::StringRef getName() const override { return "trtOpTellerPass"; }
  void runOnFunction() override;
};
}  // namespace trt
}  // namespace infrt
