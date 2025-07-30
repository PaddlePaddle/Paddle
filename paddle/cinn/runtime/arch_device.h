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
#include "paddle/cinn/runtime/backend_api.h"
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
        PADDLE_ENFORCE_EQ(
            cudaGetDevice(&device_id),
            cudaSuccess,
            ::common::errors::InvalidArgument("cudaGetDevice failed!"));
        return std::optional<int>{device_id};
#else
        return std::nullopt;
#endif
      },
      [&](common::HygonDCUArchHIP) -> std::optional<int> {
        int device_id =
            BackendAPI::get_backend(common::HygonDCUArchHIP{})->get_device();
        return std::optional<int>{device_id};
      },
      [&](common::HygonDCUArchSYCL) -> std::optional<int> {
        int device_id =
            BackendAPI::get_backend(common::HygonDCUArchSYCL{})->get_device();
        return std::optional<int>{device_id};
      });
}

void SetArchDevice(const common::Target& target,
                   const std::optional<int>& device_id) {
  target.arch.Match(
      [&](common::UnknownArch) -> void {},
      [&](common::X86Arch) -> void {},
      [&](common::ARMArch) -> void {},
      [&](common::NVGPUArch) -> void {
#ifdef CINN_WITH_CUDA
        PADDLE_ENFORCE_EQ(device_id.has_value(),
                          true,
                          ::common::errors::InvalidArgument(
                              "Required device_id should have value, but "
                              "received std::nullopt."));
        cudaSetDevice(device_id.value());
#endif
      },
      [&](common::HygonDCUArchHIP) -> void {
        PADDLE_ENFORCE_EQ(device_id.has_value(),
                          true,
                          ::common::errors::InvalidArgument(
                              "Required device_id should have value, but "
                              "received std::nullopt."));
        BackendAPI::get_backend(common::HygonDCUArchHIP{})
            ->set_device(device_id.value());
      },
      [&](common::HygonDCUArchSYCL) -> void {
        PADDLE_ENFORCE_EQ(device_id.has_value(),
                          true,
                          ::common::errors::InvalidArgument(
                              "Required device_id should have value, but "
                              "received std::nullopt."));
        BackendAPI::get_backend(common::HygonDCUArchSYCL{})
            ->set_device(device_id.value());
      });
}

}  // namespace cinn::runtime
