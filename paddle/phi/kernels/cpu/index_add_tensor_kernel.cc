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

#include "paddle/phi/kernels/index_add_tensor_kernel.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/cpu/index_add_impl.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void IndexAddTensorKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& add_tensor,
                           const IntArray& index_arr,
                           const Scalar& axis_scalar,
                           DenseTensor* output) {
  T add_value = static_cast<T>(0);
  const T* add_tensor_ptr = add_tensor.data<T>();
  paddle::memory::Copy(dev_ctx.GetPlace(),
                       &add_value,
                       dev_ctx.GetPlace(),
                       add_tensor_ptr,
                       sizeof(T));
  IndexAddBaseKernel<T, Context>(dev_ctx,
                                  x,
                                  index_arr,
                                  axis_scalar,
                                  static_cast<float>(add_value),
                                  output,
                                  nullptr);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_add_tensor,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexAddTensorKernel,
                   bool,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   double,
                   int,
                   int64_t) {}