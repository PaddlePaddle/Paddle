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

#include "paddle/phi/kernels/one_hot_kernel.h"

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

template <typename InT, typename OutT>
__global__ void FillOutputKernel(const InT* p_in_data,
                                 OutT* p_out_data,
                                 const int64_t numel,
                                 const int depth) {
  CUDA_KERNEL_LOOP_TYPE(idx, numel, int64_t) {
    PADDLE_ENFORCE(p_in_data[idx] >= 0 && p_in_data[idx] < depth,
                   "Illegal index value, Input(input) value should be "
                   "greater than or equal to 0, and less than depth [%d], "
                   "but received [%lld].",
                   depth,
                   p_in_data[idx]);

    *(p_out_data + (idx * depth) + p_in_data[idx]) = 1.0;
  }
}

template <typename T, typename Context>
void OneHotKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& depth,
                  DenseTensor* out) {
  auto depth_v = depth.to<int>();
  auto out_dims = out->dims();
  if (out_dims[out_dims.size() - 1] == -1) {
    out_dims[out_dims.size() - 1] = depth_v;
    out->Resize(out_dims);
  }

  auto* p_in_data = x.data<T>();
  auto numel = x.numel();
  auto* p_out_data = dev_ctx.template Alloc<float>(out);
  auto stream = dev_ctx.stream();
  funcs::set_constant(dev_ctx, out, static_cast<float>(0.0));

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);

  FillOutputKernel<<<config.block_per_grid,
                     config.thread_per_block,
                     0,
                     stream>>>(p_in_data, p_out_data, numel, depth_v);
}

}  // namespace phi

PD_REGISTER_KERNEL(one_hot, GPU, ALL_LAYOUT, phi::OneHotKernel, int, int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
}
