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
#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/fill_constant_op.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

namespace paddle {
namespace operators {

namespace kps = phi::kps;
template <typename T, typename Functor, int VecSize>
__global__ void VectorizedIndexKernel(T *out, size_t numel, size_t main_offset,
                                      Functor func) {
  size_t data_offset = BLOCK_ID_X * BLOCK_NUM_X * VecSize;
  size_t stride = BLOCK_NUM_X * GRID_NUM_X * VecSize;
  size_t args[VecSize];
  T result[VecSize];
  for (; data_offset < main_offset; data_offset += stride) {
    kps::InitWithDataIndex<size_t, VecSize, 1, 1>(&args[0], data_offset);
    kps::ElementwiseUnary<size_t, T, VecSize, 1, 1, Functor>(&result[0],
                                                             &args[0], func);
    kps::WriteData<T, VecSize, 1, 1, false>(out + data_offset, &result[0],
                                            BLOCK_NUM_X * VecSize);
  }
  size_t num = numel - data_offset;
  if (num > 0) {
    kps::InitWithDataIndex<size_t, VecSize, 1, 1>(&args[0], data_offset);
    kps::ElementwiseUnary<size_t, T, VecSize, 1, 1, Functor>(&result[0],
                                                             &args[0], func);
    kps::WriteData<T, VecSize, 1, 1, true>(out + data_offset, &result[0], num);
  }
}

template <typename T, typename Functor>
void IndexKernel(const KPDevice &dev_ctx, Tensor *out, Functor func) {
  int numel = out->numel();
  T *out_data = out->mutable_data<T>(dev_ctx.GetPlace());
  if (numel <= 0) return;
  int vec_size = phi::GetVectorizedSize(out_data);
#ifdef PADDLE_WITH_XPU_KP
  int block = 64;
  int grid = 8;
  auto stream = dev_ctx.x_context()->xpu_stream;
#else
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, vec_size);
  int grid = config.block_per_grid.x;
  int block = config.thread_per_block.x;
  auto stream = dev_ctx.stream();
#endif
  size_t main_offset = (numel / (vec_size * block)) * vec_size * block;
  switch (vec_size) {
    case 4:
      VectorizedIndexKernel<T, Functor, 4><<<grid, block, 0, stream>>>(
          out_data, numel, main_offset, func);
      break;
    case 2:
      VectorizedIndexKernel<T, Functor, 2><<<grid, block, 0, stream>>>(
          out_data, numel, main_offset, func);
      break;
    case 1:
      VectorizedIndexKernel<T, Functor, 1><<<grid, block, 0, stream>>>(
          out_data, numel, main_offset, func);
      break;
    default: {
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
}

}  // namespace operators
}  // namespace paddle
