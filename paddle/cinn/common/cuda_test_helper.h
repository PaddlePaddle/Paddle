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

#pragma once

#include <string>
#include <vector>

#include "paddle/cinn/backends/llvm/codegen_llvm.h"
#include "paddle/cinn/backends/llvm/simple_jit.h"
#include "paddle/cinn/cinn.h"

namespace cinn {
namespace common {

#ifdef CINN_WITH_CUDA
class CudaModuleTester {
 public:
  CudaModuleTester();

  // Call the host function in JIT.
  void operator()(const std::string& fn_name, void* args, int arg_num);

  void Compile(const ir::Module& m, const std::string& rewrite_cuda_code = "");

  void* LookupKernel(const std::string& name);

  void* CreateDeviceBuffer(const cinn_buffer_t* host_buffer);

  ~CudaModuleTester();

 private:
  std::unique_ptr<backends::SimpleJIT> jit_;

  void* stream_{};

  std::vector<void*> kernel_handles_;

  void* cuda_module_{nullptr};
};

#endif

}  // namespace common
}  // namespace cinn
