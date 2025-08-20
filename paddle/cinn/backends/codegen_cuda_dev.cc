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

#include "paddle/cinn/backends/codegen_cuda_dev.h"

namespace cinn {
namespace backends {
const std::string CodeGenCudaDev::general_source_header_ =  // NOLINT
    R"(
#pragma once
#include <cstdint>
#define CINN_WITH_CUDA
#include "bfloat16.h"
#include "float16.h"
#include "float8e4m3.h"
using cinn::common::bfloat16;
using cinn::common::float16;
using cinn::common::float8e4m3;
using cinn::common::half4;
using cinn::common::half8;
using cinn::common::float168;
using cinn::common::float164;
using cinn::common::float162;
using cinn::common::bfloat168;
using cinn::common::bfloat164;
using cinn::common::float8e4m32;
using cinn::common::float8e4m34;
using cinn::common::bfloat162;
#include <cooperative_groups.h>
#include "cinn_cuda_runtime_source.cuh"
)";
const std::string CodeGenCudaDev::source_header_ =  // NOLINT
    R"(
#pragma once
#include <cinn_with_cuda_h>

#include <bfloat16_h>
#include <cstdint>
#include <float16_h>
using cinn::common::bfloat16;
using cinn::common::float16;
using cinn::common::float8e4m3;
using cinn::common::half4;
using cinn::common::half8;
using cinn::common::float168;
using cinn::common::float164;
using cinn::common::float162;
using cinn::common::bfloat168;
using cinn::common::bfloat164;
using cinn::common::bfloat162;
using cinn::common::float8e4m3;
using cinn::common::float8e4m32;
using cinn::common::float8e4m34;
#include <cooperative_groups.h>
#include <cinn_cuda_runtime_source_h>
)";

const std::string &CodeGenCudaDev::GetSourceHeader() { return source_header_; }
const std::string &CodeGenCudaDev::GetGeneralSourceHeader() {
  return general_source_header_;
}

CodeGenCudaDev::CodeGenCudaDev(Target target) : CodeGenGpuDev(target) {}

void CodeGenCudaDev::PrintIncludes() { str_ += GetSourceHeader(); }

}  // namespace backends
}  // namespace cinn
