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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/uniform_random_functor.h"

namespace phi {
namespace sr {

template <typename T, typename Context>
void GPUUniformRandomKernel(const Context& dev_ctx,
                            const SelectedRows& input,
                            const std::vector<int>& shape,
                            int input_dim_idx,
                            int output_dim_idx,
                            float min,
                            float max,
                            int seed,
                            int diag_num,
                            int diag_step,
                            float diag_val,
                            DataType dtype UNUSED,
                            SelectedRows* out) {
  out->set_rows(input.rows());
  out->set_height(input.height());
  phi::DenseTensor* tensor = out->mutable_value();
  tensor->Resize(common::make_ddim(shape));
  dev_ctx.template Alloc<T>(tensor);
  phi::funcs::UniformRandom<T>(
      reinterpret_cast<const phi::GPUContext&>(dev_ctx),
      tensor,
      seed,
      min,
      max,
      diag_num,
      diag_step,
      diag_val);
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_KERNEL(uniform_random_batch_size_like_sr,
                   GPU,
                   ALL_LAYOUT,
                   phi::sr::GPUUniformRandomKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}
