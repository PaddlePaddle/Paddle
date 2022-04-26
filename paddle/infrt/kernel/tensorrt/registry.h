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
#include <string>

namespace infrt {
namespace host_context {

struct KernelRegistry;

}  // namespace host_context
}  // namespace infrt

namespace infrt {
namespace kernel {

/**
 * Register all the trt kernels to registry.
 */
void RegisterTrtKernels(host_context::KernelRegistry* registry);

}  // namespace kernel
}  // namespace infrt
