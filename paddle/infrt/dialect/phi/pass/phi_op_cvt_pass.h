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
#include "paddle/infrt/dialect/infrt/common_type.h"

namespace infrt {
/*
 * phiOpCvtPass.
 *
 * Convert the general operators in pd Dialect to a infrt.kernelOp.
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
 *  %c = "infrt.kernel"(%a){name = "conv2d"} ...
 *  %d = "infrt.kernel"(%c){name = "conv3d"}...
 *  %f = "infrt.kernel"(%a){name = "conv2d"}...
 *  "pd.fetch" (%d, %f)
 * }
 */
class phiOpCvtPass
    : public mlir::PassWrapper<phiOpCvtPass, mlir::FunctionPass> {
 public:
  ::llvm::StringRef getName() const override { return "phiOpCvtPass"; }
  void runOnFunction() override;
  explicit phiOpCvtPass(std::vector<Place> valid_places = std::vector<Place>())
      : valid_places_(valid_places) {}

 private:
  void convertStage();
  void diapatchStage();
  std::vector<Place> valid_places_;
};
}  // namespace infrt
