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

#include "paddle/infrt/dialect/mlir_loader.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Parser.h>

#include <string>

#include "paddle/infrt/dialect/init_infrt_dialects.h"

namespace infrt {
namespace dialect {

TEST(MlirLoader, basic) {
  mlir::MLIRContext context;

  auto source = R"ROC(
func @main() -> f32 {
  %v0 = infrt.constant.f32 1.0
  %v1 = infrt.constant.f32 2.0
  %value = "infrt.add.f32"(%v0, %v1) : (f32, f32) -> f32

  "infrt.print.f32"(%v0) : (f32) -> ()

  infrt.return %value : f32
}
)ROC";

  auto module = LoadMlirSource(&context, source);
  EXPECT_TRUE(mlir::succeeded(module->verify()));
  LOG(INFO) << "module name: " << module->getOperationName().data();
  for (auto func : module->getOps<mlir::FuncOp>()) {
    LOG(INFO) << "get func " << func.getName().str();
    int num_args = func.getNumArguments();
    for (int i = 0; i < num_args; i++) {
      LOG(INFO) << "arg: " << func.getArgument(i).getArgNumber();
    }
  }
}

}  // namespace dialect
}  // namespace infrt
