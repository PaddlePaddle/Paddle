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
namespace xpu {

/**
 * XPU device code generator.
 *
 * Generates __global__ kernels targeting XPU hardware (CUDA-compatible
 * programming model). Delegates to CodeGenGpuDev for the bulk of GPU codegen;
 * only the source preamble and include header differ from the NVGPU path.
 */
class CodeGenXpuDevice : public CodeGenGpuDev {
 public:
  explicit CodeGenXpuDevice(Target target);
  static const std::string& GetSourceHeader();
  void PrintIncludes() override;

 private:
  static const std::string source_header_;
};

}  // namespace xpu
}  // namespace backends
}  // namespace cinn
