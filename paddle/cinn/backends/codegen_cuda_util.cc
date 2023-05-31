// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "cinn/backends/codegen_cuda_util.h"

#include "cinn/backends/cuda_util.h"
#include "cinn/ir/ir_mutator.h"

namespace cinn {
namespace backends {

std::tuple<ir::Module, ir::Module> SplitCudaAndHostModule(ir::Module module) {
  detail::CollectHostFunctionVisitor visitor(module->name);
  Expr expr(module);
  return visitor(&expr);
}

}  // namespace backends
}  // namespace cinn
