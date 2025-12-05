// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/random_kernel.h"

#include <random>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"

namespace phi {
template <typename T, typename Context>
void RandomKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int64_t from,
                  int64_t to,
                  DenseTensor* out) {
  out->Resize(x.dims());
  T* data = dev_ctx.template Alloc<T>(out);

  if constexpr (std::is_floating_point_v<T> ||
                std::is_same_v<T, phi::float16> ||
                std::is_same_v<T, phi::bfloat16>) {
    from = update_from<T>(from);
    to = update_to<T>(to);

    PADDLE_ENFORCE_LT(from,
                      to,
                      common::errors::InvalidArgument(
                          "random expects 'from' casted to dtype to be less "
                          "than 'to' casted to dtype, but got from=%d >= to=%d",
                          from,
                          to));
  }
  uint64_t range = static_cast<uint64_t>(to) - static_cast<uint64_t>(from);
  if (range >= 1ULL << 28) {
    funcs::uniform_distribution<uint64_t> dist;
    funcs::uniform_int_from_to_distribution<T, uint64_t> random(range, from);
    funcs::distribution_and_transform<T>(dev_ctx, out, dist, random);

  } else {
    funcs::uniform_distribution<uint32_t> dist;
    funcs::uniform_int_from_to_distribution<T, uint32_t> random(range, from);
    funcs::distribution_and_transform<T>(dev_ctx, out, dist, random);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(random,
                   GPU,
                   ALL_LAYOUT,
                   phi::RandomKernel,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
