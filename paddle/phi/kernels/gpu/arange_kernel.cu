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

#include "paddle/phi/kernels/arange_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/range_function.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/bfloat16.h"

namespace phi {

template <typename T>
__global__ void Range(T start, T step, int64_t size, T* out) {
  CUDA_KERNEL_LOOP(index, size) { out[index] = start + step * index; }
}

template <typename T, typename Context>
void ArangeKernel(const Context& dev_ctx,
                  const DenseTensor& start,
                  const DenseTensor& end,
                  const DenseTensor& step,
                  DenseTensor* out) {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType start_value = static_cast<MPType>(GetValue<T, Context>(dev_ctx, start));
  MPType end_value = static_cast<MPType>(GetValue<T, Context>(dev_ctx, end));
  MPType step_value = static_cast<MPType>(GetValue<T, Context>(dev_ctx, step));

  int64_t size = 0;
  
  phi::funcs::GetSize(start_value, end_value, step_value, &size);
  out->Resize(phi::make_ddim({size}));
  MPType* out_data = dev_ctx.template Alloc<MPType>(out);

  auto stream = dev_ctx.stream();
  int64_t block = std::min(size, static_cast<int64_t>(256));
  if (block == 0) {
    return;
  }
  int64_t grid = (size + block - 1) / block;
  Range<MPType><<<grid, block, 0, stream>>>(start_value, step_value, size, out_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    arange, GPU, ALL_LAYOUT, phi::ArangeKernel, float, double, int64_t, int, phi::dtype::float16, phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
}
