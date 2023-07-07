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

#include "paddle/fluid/operators/cinn/cinn_launch_op.h"

#include "paddle/cinn/runtime/flags.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/core/generator.h"

namespace paddle {
namespace operators {

namespace details {

template <>
void SetCinnRandomSeed<phi::GPUContext>() {
  auto seed = phi::DefaultCUDAGenerator(0)->GetCurrentSeed();
  ::cinn::runtime::RandomSeed::GetOrSet(seed);
}

}  // namespace details
}  // namespace operators
}  // namespace paddle

/* see [Why use single type kernel] */
PD_REGISTER_STRUCT_KERNEL(cinn_launch,
                          GPU,
                          ALL_LAYOUT,
                          paddle::operators::CinnLaunchOpKernel,
                          float) {}
