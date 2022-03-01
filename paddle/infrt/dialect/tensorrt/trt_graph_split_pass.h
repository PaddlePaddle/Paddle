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
 * trtGraphSplitPass.
 *
 * Splite the graph op when the number of operations is too small.
 * The feature is the opposite of 'trtOpTellerPass'.
 *
 * source func:
 *
 * func @main() -> tensor<?xf32> {
 *  %a = "pd.feed"()...
 *  %d, %f = "trt.create_engine"(%a) {
 *     %m = "pd.conv2d"(%a)...
 *     %n = "pd.conv3d"(%m)...
 *     %s = "pd.conv2d"(%a)...
 *     "Infrt.return" (%n, %s)
 *  } ...
 *  "pd.fetch" (%d, %f)
 * }
 *
 * destination func:
 * func @main() -> tensor<?xf32> {
 *  %a = "pd.feed"()...
 *  %c = "pd.conv2d"(%a) ...
 *  %d = "pd.conv3d"(%c) ...
 *  %f = "pd.conv2d"(%a) ...
 *  "pd.fetch" (%d, %f)
 * }
 */
class TRTGraphSplitPass
    : public mlir::PassWrapper<TRTGraphSplitPass, mlir::FunctionPass> {
 public:
  ::llvm::StringRef getName() const override { return "trtGraphSplitPass"; }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<TensorRTDialect, ::infrt::dialect::INFRTDialect>();
  }
  void runOnFunction() override;
  explicit TRTGraphSplitPass(size_t min_subgraph_size = 3)
      : min_subgraph_size_(min_subgraph_size) {}

 private:
  size_t min_subgraph_size_;
};
}  // namespace trt
}  // namespace infrt
