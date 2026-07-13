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

#include "paddle/cinn/backends/xpu/codegen_xpu_dev.h"

namespace cinn {
namespace backends {
namespace xpu {

// XPU uses the CUDA programming model: include the CUDA-compatible
// cinn_xpu_runtime_source.cuh device header.
const std::string CodeGenXpuDevice::source_header_ =  // NOLINT
    R"(#define CINN_WITH_XPU
     #include "float16.h"
     using cinn::common::float16;
     #include "bfloat16.h"
     using cinn::common::bfloat16;
     #include "cinn_xpu_runtime_source.cuh"
)";

const std::string& CodeGenXpuDevice::GetSourceHeader() {
  return source_header_;
}

CodeGenXpuDevice::CodeGenXpuDevice(Target target) : CodeGenGpuDev(target) {}

void CodeGenXpuDevice::PrintIncludes() { str_ += GetSourceHeader(); }

}  // namespace xpu
}  // namespace backends
}  // namespace cinn
