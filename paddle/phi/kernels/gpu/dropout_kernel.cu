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

#include "paddle/phi/kernels/dropout_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/dropout_impl.cu.h"
#include "paddle/phi/kernels/funcs/hash_utils.h"
#include "cuda_runtime.h"
#include <xxhash.h>

namespace phi {

template <typename T, typename Context>
int64_t hash_tensor(const Context& dev_ctx, phi::DenseTensor print_tensor) {
  const T* data = nullptr;
  phi::DenseTensor cpu_tensor;
  if (print_tensor.place().GetType() == phi::AllocationType::CPU) {
    data = print_tensor.data<T>();
  } else {
    phi::CPUPlace cpu_place;
    phi::Copy(dev_ctx, print_tensor, cpu_place, false, &cpu_tensor);
    data = cpu_tensor.data<T>();
  }

  return static_cast<int64_t>(XXH64(data, sizeof(T) * cpu_tensor.numel(), 0));
}

template <typename T, typename Context>
void DropoutRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const paddle::optional<DenseTensor>& seed_tensor,
                      const Scalar& p,
                      bool is_test,
                      const std::string& mode,
                      int seed,
                      bool fix_seed,
                      DenseTensor* out,
                      DenseTensor* mask) {

  // std::cout<<"seed "<<seed<<std::endl;
  
  // auto x_hash_value = hash_tensor<T, Context>(dev_ctx, x);
  // std::cout<<"before dropout hash value:  "<<x_hash_value<<std::endl;
  
  bool upscale_in_train = (mode == "upscale_in_train");
  dev_ctx.template Alloc<T>(out);
  if (mask) {
    dev_ctx.template Alloc<uint8_t>(mask);
  }
  phi::funcs::DropoutFwGPUKernelDriver<T>(dev_ctx,
                                          is_test,
                                          p.to<float>(),
                                          upscale_in_train,
                                          fix_seed,
                                          seed,
                                          x,
                                          seed_tensor.get_ptr(),
                                          mask,
                                          out);

  // auto out_hash_value = hash_tensor<T, Context>(dev_ctx, *out);
  // std::cout<<"after dropout hash value:  "<<out_hash_value<<std::endl;
}

template <typename T, typename Context>
void DropoutNdKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& seed_tensor,
                     const Scalar& p,
                     bool is_test,
                     const std::string& mode,
                     int seed,
                     bool fix_seed,
                     const std::vector<int>& axis,
                     DenseTensor* out,
                     DenseTensor* mask) {
  bool upscale_in_train = (mode == "upscale_in_train");
  dev_ctx.template Alloc<T>(out);
  if (mask) {
    dev_ctx.template Alloc<uint8_t>(mask);
  }
  phi::funcs::DropoutFwGPUKernelDriver<T>(dev_ctx,
                                          is_test,
                                          p.to<float>(),
                                          upscale_in_train,
                                          fix_seed,
                                          seed,
                                          x,
                                          seed_tensor.get_ptr(),
                                          mask,
                                          out,
                                          true,
                                          axis);
}

}  // namespace phi

PD_REGISTER_KERNEL(dropout,
                   GPU,
                   ALL_LAYOUT,
                   phi::DropoutRawKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(1).SetDataType(phi::DataType::UINT8);
}

PD_REGISTER_KERNEL(dropout_nd,
                   GPU,
                   ALL_LAYOUT,
                   phi::DropoutNdKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(1).SetDataType(phi::DataType::UINT8);
}
