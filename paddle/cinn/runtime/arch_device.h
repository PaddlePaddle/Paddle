// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/target.h"

namespace cinn::runtime {

int GetArchDevice(const common::Target& target) {
  return target.arch.Visit(
      adt::match{[&](common::UnknownArch) -> int { CINN_NOT_IMPLEMENTED; },
                 [&](common::X86Arch) -> int { CINN_NOT_IMPLEMENTED; },
                 [&](common::ARMArch) -> int { CINN_NOT_IMPLEMENTED; },
                 [&](common::NVGPUArch) -> int {
#ifdef CINN_WITH_CUDA
                   int device_id;
                   cudaGetDevice(&device_id);
                   return device_id;
#else
                   CINN_NOT_IMPLEMENTED
#endif
                 }});
}

void SetArchDevice(const common::Target& target, int device_id) {
  target.arch.Visit(
      adt::match{[&](common::UnknownArch) -> void { CINN_NOT_IMPLEMENTED; },
                 [&](common::X86Arch) -> void { CINN_NOT_IMPLEMENTED; },
                 [&](common::ARMArch) -> void { CINN_NOT_IMPLEMENTED; },
                 [&](common::NVGPUArch) -> void {
#ifdef CINN_WITH_CUDA
                   cudaSetDevice(device_id);
#else
                   CINN_NOT_IMPLEMENTED
#endif
                 }});
}

}  // namespace cinn::runtime
