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

#include "paddle/infrt/kernel/control_flow_kernels.h"

#include <glog/logging.h>

#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/mlir_function_executable.h"

namespace infrt {
namespace kernel {

static void INFRTCall(
    host_context::RemainingArguments args,
    host_context::RemainingResults results,
    host_context::Attribute<host_context::MlirFunctionExecutable*> fn) {
  VLOG(3) << "running call kernel ...";
  CHECK_EQ(fn.get()->num_arguments(), args.size());
  CHECK_EQ(fn.get()->num_results(), results.size());

  for (auto& v : results.values()) {
    CHECK(v.get());
  }
  fn.get()->Execute(args.values(), results.values());
}

void RegisterControlFlowKernels(host_context::KernelRegistry* registry) {
  registry->AddKernel("infrt.call", INFRT_KERNEL(INFRTCall));
}

}  // namespace kernel
}  // namespace infrt
