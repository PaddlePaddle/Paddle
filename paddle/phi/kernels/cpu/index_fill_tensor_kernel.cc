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

#include "paddle/phi/kernels/index_fill_tensor_kernel.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/cpu/index_fill_impl.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void IndexFillTensorKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& fill_tensor,
                           const IntArray& index_arr,
                           const Scalar& axis_scalar,
                           DenseTensor* output) {
  T fill_value = static_cast<T>(0);
  const T* fill_tensor_ptr = fill_tensor.data<T>();
  paddle::memory::Copy(phi::CPUPlace(),
                       &fill_value,
                       dev_ctx.GetPlace(),
                       fill_tensor_ptr,
                       sizeof(T));
  IndexFillBaseKernel<T, Context>(dev_ctx,
                                  x,
                                  index_arr,
                                  axis_scalar,
                                  static_cast<float>(fill_value),
                                  output,
                                  nullptr);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_fill_tensor,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexFillTensorKernel,
                   bool,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   double,
                   int,
                   int64_t) {}
