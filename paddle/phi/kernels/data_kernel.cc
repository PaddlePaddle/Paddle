// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/data_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/data_impl.h"

#include "paddle/phi/kernels/funcs/tensor_formatter.h"

namespace phi {

template <typename T, typename Context>
void DataKernel(const Context& ctx,
                const std::string& name,
                const phi::IntArray& shape,
                phi::DataType data_type,
                DenseTensor* out) {}

template <typename T, typename Context>
void ShadowOutputKernel(const Context& ctx,
                        const DenseTensor& x,
                        DenseTensor* out) {}

}  // namespace phi

PD_REGISTER_KERNEL(data,
                   CPU,
                   ALL_LAYOUT,
                   phi::DataKernel,
                   bool,
                   uint8_t,
                   float,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(shadow_feed,
                                         ALL_LAYOUT,
                                         phi::ShadowFeedKernel) {}
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(shadow_feed_tensors,
                                         ALL_LAYOUT,
                                         phi::ShadowFeedTensorsKernel) {}
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(print_kernel,
                                         ALL_LAYOUT,
                                         phi::PrintKernel) {}
PD_REGISTER_KERNEL(
    shadow_output, CPU, ALL_LAYOUT, phi::ShadowOutputKernel, float) {}
