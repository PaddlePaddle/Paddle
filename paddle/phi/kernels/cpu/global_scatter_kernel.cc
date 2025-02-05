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

#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GlobalScatterKernel(const Context &dev_ctx UNUSED,
                         const DenseTensor &x UNUSED,
                         const DenseTensor &local_count UNUSED,
                         const DenseTensor &global_count UNUSED,
                         DenseTensor *out UNUSED) {
  PADDLE_THROW(common::errors::Unavailable(
      "Do not support global scatter op for cpu kernel now."));
}

}  // namespace phi

PD_REGISTER_KERNEL(global_scatter,
                   CPU,
                   ALL_LAYOUT,
                   phi::GlobalScatterKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
