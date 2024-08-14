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

#include "paddle/cinn/backends/hip/codegen_hip_dev.h"

namespace cinn {
namespace backends {
namespace hip {

const std::string CodeGenHipDevice::source_header_ =  // NOLINT
    R"(#include "cinn_hip_runtime_source.h"
)";

const std::string &CodeGenHipDevice::GetSourceHeader() {
  return source_header_;
}

CodeGenHipDevice::CodeGenHipDevice(Target target) : CodeGenGpuDev(target) {}

void CodeGenHipDevice::PrintIncludes() { str_ += GetSourceHeader(); }

}  // namespace hip
}  // namespace backends
}  // namespace cinn
