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
#include "paddle/common/enforce.h"

namespace cinn::runtime {

std::optional<int> GetArchDevice(const common::Target& target) {
  return target.arch.Match(
      [&](common::UnknownArch) -> std::optional<int> { return std::nullopt; },
      [&](common::X86Arch) -> std::optional<int> { return std::nullopt; },
      [&](common::ARMArch) -> std::optional<int> { return std::nullopt; },
      [&](common::NVGPUArch) -> std::optional<int> {
#ifdef CINN_WITH_CUDA
        int device_id;
        cudaGetDevice(&device_id);
        return std::optional<int>{device_id};
#else
        return std::nullopt;
#endif
      });
}

void SetArchDevice(const common::Target& target, int device_id) {
  PADDLE_ENFORCE_GE(device_id,
                    0,
                    ::common::errors::InvalidArgument(
                        "Required device_id >=0  but received %d.", device_id));
  target.arch.Match([&](common::UnknownArch) -> void {},
                    [&](common::X86Arch) -> void {},
                    [&](common::ARMArch) -> void {},
                    [&](common::NVGPUArch) -> void {
#ifdef CINN_WITH_CUDA
                      cudaSetDevice(device_id);
#endif
                    });
}

}  // namespace cinn::runtime
