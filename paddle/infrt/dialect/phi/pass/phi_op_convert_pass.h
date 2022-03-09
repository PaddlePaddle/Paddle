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
class PhiOpConvertPass
    : public mlir::PassWrapper<PhiOpConvertPass, mlir::FunctionPass> {
 public:
  ::llvm::StringRef getName() const override { return "phi-op-convert"; }
  void runOnFunction() override;

  PhiOpConvertPass();

  explicit PhiOpConvertPass(const std::vector<Place>& valid_places)
      : valid_places_(valid_places) {}

  PhiOpConvertPass(const PhiOpConvertPass& other)
      : mlir::PassWrapper<PhiOpConvertPass, mlir::FunctionPass>(*this),
        valid_places_(other.valid_places_) {}

  llvm::StringRef getArgument() const override { return "phi-op-convert"; }

  void getDependentDialects(mlir::DialectRegistry& registry) const override;

 private:
  void convertStage();
  void dispatchStage();

  // Force a specified data format for all layout sensitive operations.
  Option<std::string> valid_places_options_{
      *this,
      "valid-targets",
      llvm::cl::desc("Set the valid target, [CPU-FP32-NCHW]")};

  std::vector<Place> valid_places_;
};
}  // namespace infrt
