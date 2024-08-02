// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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
#include "paddle/cinn/backends/codegen_gpu_dev.h"

namespace cinn {
namespace backends {

/**
 * CUDA device code generator.
 *
 * It generates the device function, e.g, the function called "myadd" will have
 * a __global__ function called "myadd_kernel", different from codegen_c, the
 * declaration of the "myadd_kernel" function has an expanded argument list,
 * which finally similar to `__global__ void myadd(float* __restrict__ A, float*
 * __restrict__ B, int n);`
 */
class CodeGenCudaDev : public CodeGenGpuDev {
 public:
  explicit CodeGenCudaDev(Target target);
  static const std::string& GetSourceHeader();
  void PrintIncludes() override;

 private:
  static const std::string source_header_;
};

}  // namespace backends
}  // namespace cinn
