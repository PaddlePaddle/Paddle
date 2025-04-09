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

#include "paddle/cinn/common/arch_util.h"

namespace cinn {
namespace common {

std::string GetArchNameImpl(UnknownArch arch) { return "Unk"; }

std::string GetArchNameImpl(X86Arch arch) { return "X86"; }

std::string GetArchNameImpl(ARMArch arch) { return "ARM"; }

std::string GetArchNameImpl(NVGPUArch arch) { return "NVGPU"; }

std::string GetArchNameImpl(HygonDCUArchHIP arch) { return "HygonDCU_HIP"; }

std::string GetArchNameImpl(HygonDCUArchSYCL arch) { return "HygonDCU_SYCL"; }

std::string GetArchName(Arch arch) {
  return std::visit([](const auto& impl) { return GetArchNameImpl(impl); },
                    arch.variant());
}

std::ostream& operator<<(std::ostream& os, Arch arch) {
  os << GetArchName(arch);
  return os;
}

}  // namespace common
}  // namespace cinn
