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

#include "paddle/phi/kernels/sparse/pool_grad_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void MaxPoolGradKernel(const Context& dev_ctx,
                       const SparseCooTensor& x,
                       const DenseTensor& rulebook,
                       const SparseCooTensor& out,
                       const DenseTensor& out_grad,
                       const std::vector<int>& kernel_sizes,
                       DenseTensor* x_grad) {
  /*
int kernel_size = kernel_sizes[0] * kernel_sizes[1] * kernel_sizes[2];
const int in_channels = kernel_sizes[3];
int rulebook_len = rulebook.dims()[1];
const int *rulebook_ptr = rulebook.data<int>();
std::vector<int> offsets(kernel_size + 1), counter(kernel_size, 0);
for (int i = 0; i < rulebook_len; i++) {
  counter[rulebook_ptr[i]] += 1;
}
int offset = 0;
for (int i = 0; i < kernel_size; i++) {
  offsets[i] = offset;
  offset += counter[i];
}
offsets[kernel_size] = offset;

const T* in_features_ptr = x.non_zero_elements().data<T>();
const T* out_features_ptr = out.non_zero_elements().data<T>();
const T* out_grad_ptr = out_grad.data<T>();
T* x_grad_ptr = x_grad->data<T>();

for(int i = 0; i < kernel_size; i++){
    if(counter[i] <= 0){
        continue;
    }

    for(int j = 0; j < counter[i]; j++){
        int in_i = rulebook_ptr[rulebook_len + offsets[i] + j];
        int out_i = rulebook_ptr[rulebook_len*2 + offsets[i] + j];
        for(int c = 0; c < in_channels; c++){
          if(out_features_ptr[out_i * in_channels + c] == in_features_ptr[in_i *
in_channels + c]){
              x_grad_ptr[in_i * in_channels + c] = out_grad_ptr[out_i *
in_channels + c];
          }
        }
    }
}
*/
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sparse_maxpool_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::MaxPoolGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
