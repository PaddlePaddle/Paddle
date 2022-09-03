// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/backends/host/phi_allocator.h"
#include "paddle/infrt/backends/host/phi_context.h"
#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/phi/core/dense_tensor.h"

namespace infrt {
namespace kernel {
namespace phi {

::phi::CPUContext CreateCPUContext();

#ifdef INFRT_WITH_GPU
::phi::GPUContext CreateGPUContext();
#endif

}  // namespace phi
}  // namespace kernel
}  // namespace infrt
