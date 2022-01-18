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

#include <mlir/Support/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>
#include "paddle/infrt/dialect/init_infrt_dialects.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  infrt::registerCinnDialects(registry);
  mlir::registerCanonicalizerPass();
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "infrt mlir pass driver", registry));
}
