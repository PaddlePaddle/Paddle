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

#include "paddle/phi/kernels/index_add_kernel.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
// #include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/cpu/index_add_impl.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void IndexAddKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& index,
                    const DenseTensor& add_value,
                    int axis,
                    DenseTensor* output) {
  IndexAddBaseKernel<T, Context>(dev_ctx, x, index, axis, add_value, output);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_add,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexAddKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t) {}
