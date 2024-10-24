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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/uniform_random_functor.h"

namespace phi {
namespace sr {

template <typename T, typename Context>
void CPUUniformRandomKernel(const Context& dev_ctx,
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
  T* data = dev_ctx.template Alloc<T>(tensor);
  int64_t size = tensor->numel();

  phi::funcs::UniformRealDistribution<T>(
      data, size, min, max, static_cast<unsigned int>(seed));

  unsigned int diag_num_tmp = static_cast<unsigned int>(diag_num);
  unsigned int diag_step_tmp = static_cast<unsigned int>(diag_step);
  auto diag_val_tmp = static_cast<T>(diag_val);
  if (diag_num_tmp > 0) {
    PADDLE_ENFORCE_GT(
        size,
        (diag_num_tmp - 1) * (diag_step_tmp + 1),
        common::errors::InvalidArgument(
            "ShapeInvalid: the diagonal's elements is equal (num-1) "
            "* (step-1) with num %d, step %d,"
            "It should be smaller than %d, but received %d",
            diag_num_tmp,
            diag_step_tmp,
            (diag_num_tmp - 1) * (diag_step_tmp + 1),
            size));
    for (int64_t i = 0; i < diag_num_tmp; ++i) {
      int64_t pos = i * diag_step_tmp + i;
      data[pos] = diag_val_tmp;
    }
  }
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_KERNEL(uniform_random_batch_size_like_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::CPUUniformRandomKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}
