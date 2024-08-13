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
#include "paddle/phi/kernels/funcs/uniform_random_functor.h"
namespace phi {

template <typename T, typename Context>
void GPUUniformRandomKernel(const Context& dev_ctx,
                            const DenseTensor& input,
                            const std::vector<int>& shape,
                            int input_dim_idx,
                            int output_dim_idx,
                            float min,
                            float max,
                            int seed,
                            int diag_num,
                            int diag_step,
                            float diag_val,
                            DataType dtype,
                            DenseTensor* out) {
  phi::funcs::UniformRandom<T>(
      dev_ctx, out, seed, min, max, diag_num, diag_step, diag_val);
}
}  // namespace phi

PD_REGISTER_KERNEL(uniform_random_batch_size_like,
                   GPU,
                   ALL_LAYOUT,
                   phi::GPUUniformRandomKernel,
                   float,
                   double) {}
