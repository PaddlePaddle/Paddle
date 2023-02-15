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

#include <string>
#include <tuple>
#include <utility>

#include "mlir/IR/Operation.h"
#include "paddle/infrt/backends/tensorrt/trt_engine.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

namespace infrt {
namespace host_context {
class SymbolTable;
}  // namespace host_context

namespace kernel {
namespace tensorrt {

struct MlirOperationWithInfrtSymbol {
  mlir::Operation* operation;
  ::infrt::host_context::SymbolTable* symbol_table;
};

::infrt::backends::tensorrt::TrtEngine CreateTrtEngine(
    MlirOperationWithInfrtSymbol engine_op);

void PrintTrtLayer(backends::tensorrt::TrtEngine* engine);

std::vector<::Tensor*> TrtEngineCompute(backends::tensorrt::TrtEngine* engine,
                                        const ::phi::GPUContext& context);

}  // namespace tensorrt
}  // namespace kernel
}  // namespace infrt
