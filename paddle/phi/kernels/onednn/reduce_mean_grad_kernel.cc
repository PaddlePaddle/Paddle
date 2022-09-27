/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/reduce_mean_grad_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/onednn/reduce_kernel_impl.h"

namespace phi {
template <typename T, typename Context>
void MeanGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    const IntArray& dims,
                    bool keep_dim,
                    bool reduce_all,
                    DenseTensor* x_grad) {
  auto input_dims = phi::vectorize(x.dims());
  std::vector<int64_t> reduce_dims = dims.GetData();
  int number_of_elements = 1;
  if (reduce_all == false) {
    for (size_t i = 0; i < dims.size(); ++i) {
      reduce_dims[i] = (reduce_dims[i] >= 0)
                           ? reduce_dims[i]
                           : input_dims.size() + reduce_dims[i];
      number_of_elements *= input_dims[reduce_dims[i]];
    }
  } else {
    number_of_elements = x.numel();
  }
  const IntArray new_dims = IntArray(reduce_dims);
  ReduceGradKernel<T, Context>(dev_ctx,
                               x,
                               out_grad,
                               new_dims,
                               keep_dim,
                               reduce_all,
                               x_grad,
                               dnnl::algorithm::binary_add,
                               dnnl::algorithm::reduction_mean,
                               0.0f,
                               1.0L / number_of_elements);
}
}  // namespace phi

PD_REGISTER_KERNEL(mean_grad,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::MeanGradKernel,
                   float,
                   phi::dtype::bfloat16) {}
