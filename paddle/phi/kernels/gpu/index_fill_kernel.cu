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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/index_fill_funcs.h"
#include "paddle/phi/kernels/index_fill_kernel.h"

namespace phi {

template <typename T, typename Context>
void IndexFillKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const IntArray& index_arr,
                     const Scalar& axis_scalar,
                     float fill_value,
                     DenseTensor* output) {
  IndexFillBaseKernel<T, Context>(
      dev_ctx, x, index_arr, axis_scalar, fill_value, output, nullptr);

  /*
phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, output);


auto axis = axis_scalar.to<int>();

auto index_list = index_arr.GetData();
int64_t index_size = static_cast<int64_t>(index_list.size());
DenseTensor index;
index.Resize(make_ddim({index_size}));
int64_t* index_ptr = dev_ctx.template Alloc<int64_t>(&index);
paddle::memory::Copy(
dev_ctx.GetPlace(), index_ptr, phi::CPUPlace(), index_list.data(), index_size *
sizeof(int64_t), dev_ctx.stream());

index_fill_cuda_impl<T, Context>(dev_ctx, index, axis, fill_value, output);*/
}

}  // namespace phi

PD_REGISTER_KERNEL(index_fill,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexFillKernel,
                   bool,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   double,
                   int,
                   int64_t) {}
