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

#include "paddle/phi/kernels/split_and_relocate_kernel.h"

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"
#include "paddle/phi/kernels/split_kernel.h"

namespace phi {
template <typename T, typename Context>
void SplitAndRelocateKernel(const Context& dev_ctx,
                            const DenseTensor& concated_input,
                            const std::vector<const DenseTensor*>& input,
                            std::vector<DenseTensor*> output) {
  PADDLE_ENFORCE_EQ(input.size(),
                    output.size(),
                    errors::InvalidArgument(
                        "The number of ConcatAndRelocate operator's input and "
                        "output is not match, "
                        "input number is %u, output number is %u.",
                        input.size(),
                        output.size()));
  auto axis_val = phi::funcs::ComputeAxis(-1, concated_input.dims().size());
  SplitWithNumKernel<T, Context>(
      dev_ctx, concated_input, output.size(), axis_val, output);
}

}  // namespace phi

PD_REGISTER_KERNEL(split_and_relocate,
                   CPU,
                   ALL_LAYOUT,
                   phi::SplitAndRelocateKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(split_and_relocate,
                   GPU,
                   ALL_LAYOUT,
                   phi::SplitAndRelocateKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
