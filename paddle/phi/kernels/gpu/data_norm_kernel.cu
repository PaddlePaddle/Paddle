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

#include <memory>
#include <string>

#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

namespace phi {

static inline int GET_BLOCKS(const int N) {
  return (N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
}

template <typename T>
__global__ void KernelDataNormFF(
    int N, int C, const T *x, T *y, const T *mean, const T *scale) {
  CUDA_KERNEL_LOOP(i, N * C) {
    int col = i % C;
    y[i] = (x[i] - mean[col]) * scale[col];
  }
}

template <typename T>
__global__ void KernelMeanScale(int C,
                                const T *batch_size,
                                const T *batch_sum,
                                const T *batch_square_sum,
                                T *mean,
                                T *scale) {
  CUDA_KERNEL_LOOP(i, C) {
    mean[i] = batch_sum[i] / batch_size[i];
    scale[i] = sqrt(batch_size[i] / batch_square_sum[i]);
  }
}

template <typename T, typename Context>
void DataNormKernel(const Context &dev_ctx,
                    const paddle::optional<DenseTensor> &scale_w_in,
                    const paddle::optional<DenseTensor> &bias_in,
                    const DenseTensor &x_in,
                    const DenseTensor &batch_size,
                    const DenseTensor &batch_sum,
                    const DenseTensor &batch_square_sum,
                    float epsilon,
                    int slot_dim,
                    float summary_decay_rate,
                    bool enable_scale_and_shift,
                    const std::string &data_layout_in,
                    bool sync_stats,
                    DenseTensor *out,
                    DenseTensor *means,
                    DenseTensor *scales) {
  const auto *x = &x_in;
  const auto &x_dims = x->dims();
  // Align with CPU version, but should we add this restriction?
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      2,
      common::errors::PreconditionNotMet("The Input dim size should be 2"));
  const int N = x_dims[0];
  const int C = x_dims[1];

  PADDLE_ENFORCE_LT(0,
                    N,
                    common::errors::InvalidArgument(
                        "The dims of Input(X) should be greater than 0."));
  PADDLE_ENFORCE_LT(0,
                    C,
                    common::errors::InvalidArgument(
                        "The dims of Input(X) should be greater than 0."));

  const T *batch_size_in = batch_size.data<T>();
  const T *batch_sum_in = batch_sum.data<T>();
  const T *batch_square_sum_in = batch_square_sum.data<T>();
  auto *x_data = x->data<T>();

  // alloc memory
  T *y_data = dev_ctx.template Alloc<T>(out);
  T *mean_out_data = dev_ctx.template Alloc<T>(means);
  T *scale_out_data = dev_ctx.template Alloc<T>(scales);

  auto stream = dev_ctx.stream();

  KernelMeanScale<<<GET_BLOCKS(C), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
      C,
      batch_size_in,
      batch_sum_in,
      batch_square_sum_in,
      mean_out_data,
      scale_out_data);
  KernelDataNormFF<<<GET_BLOCKS(C * N), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
      N, C, x_data, y_data, mean_out_data, scale_out_data);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    data_norm, GPU, ALL_LAYOUT, phi::DataNormKernel, float, double) {}
