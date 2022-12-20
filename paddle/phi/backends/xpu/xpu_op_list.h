/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_XPU

#include <string>
#include <unordered_map>
#include <unordered_set>
#include "paddle/phi/common/data_type.h"

namespace phi {
namespace backends {
namespace xpu {

using XPUKernelSet = std::unordered_set<phi::DataType>;
using XPUOpMap = std::unordered_map<std::string, XPUKernelSet>;

XPUOpMap& GetKL1Ops();
XPUOpMap& GetKL2Ops();

// Since the Kunlun chip needs to compare the accuracy with the calculation
// results of the CPU during the test process, manually select an operator
// to use the CPU or use the XPU to assist in correctness analysis
bool IsInXPUBlackList(const std::string& kernel_name);

// Because there will be some differences between versions of the Kunlun chip,
// during the chip upgrade process, a certain operator Kunlun2 has it,
// but Kunlun3 has not yet supported it.
// At this time, we need to make a selection judgment through the list,
// which is mainly used in the transition stage.
bool IsXPUSupportKernel(const std::string& kernel_name,
                        phi::DataType kernel_dtype);

// ue to the above two special situations, whether the XPU falls back
// to the CPU needs to be judged separately compared to other devices
bool IsXPUFallbackToCPU(const std::string& kernel_name,
                        phi::DataType kernel_dtype,
                        bool kernel_not_exist);

}  // namespace xpu
}  // namespace backends
}  // namespace phi
#endif
