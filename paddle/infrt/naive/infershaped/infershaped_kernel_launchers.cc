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

#include "paddle/infrt/naive/infershaped/infershaped_kernel_launchers.h"
#include "paddle/infrt/naive/infershaped/elementwise_add.h"
#include "paddle/infrt/naive/infershaped/infershaped_registry.h"

namespace infrt {
namespace naive {

using ElementwiseAddLauncher =
    KernelLauncher<decltype(&ElementwiseAdd),
                   &ElementwiseAdd,
                   decltype(&ElementwiseAddInferShape),
                   &ElementwiseAddInferShape>;

void RegisterInferShapeLaunchers(InferShapedKernelRegistry* registry) {
  registry->AddKernel("elementwise_add",
                      INFERSHAPED_KERNEL_CREATOR(ElementwiseAddLauncher));
}

}  // namespace naive
}  // namespace infrt
