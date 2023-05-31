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

#include <gtest/gtest.h>
#include <stdlib.h>

#include <fstream>
#include <tuple>
#include <vector>

#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/codegen_cuda_host.h"
#include "cinn/backends/codegen_cuda_util.h"
#include "cinn/backends/extern_func_jit_register.h"
#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/cinn.h"
#include "cinn/common/ir_util.h"
#include "cinn/common/test_helper.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/lang/lower.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/utils/timer.h"

namespace cinn {
namespace backends {

TEST(CUDAFile, Module_output) {
  std::string cuda_source_name = "_generated1.cu";
  std::string cuda_source_code = R"ROC(
extern "C" {

__global__
void __launch_bounds__(200) elementwise_mul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
  if (((int)blockIdx.x < 100)) {
    if (((int)threadIdx.x < 200)) {
      C[((200 * (int)blockIdx.x) + (int)threadIdx.x)] = (A[((200 * (int)blockIdx.x) + (int)threadIdx.x)] * B[((200 * (int)blockIdx.x) + (int)threadIdx.x)]);
    };
  };
}

}
  )ROC";
  std::ofstream file(cuda_source_name);
  CHECK(file.is_open()) << "failed to open file " << cuda_source_name;
  file << CodeGenCUDA_Dev::GetSourceHeader();
  file << cuda_source_code;
  file.close();
  LOG(WARNING) << "Output C source to file " << cuda_source_name;
}

}  // namespace backends
}  // namespace cinn
