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
__global__ void KernelDataNormBP(
    int N, int C, const T *y_grad, const T *scale, T *x_grad) {
  CUDA_KERNEL_LOOP(i, N * C) { x_grad[i] = y_grad[i] * scale[i % C]; }
}

template <typename T>
__global__ void KernelDataNormBPStat(int N,
                                     int C,
                                     const T *x_val,
                                     const T *means,
                                     const float squared_sum_epsilon,
                                     T *batch_size,
                                     T *batch_sum,
                                     T *batch_square_sum) {
  CUDA_KERNEL_LOOP(i, C) {
    T val_sum = 0;
    T square_sum = 0;
    for (int j = 0; j < N; j++) {
      val_sum += x_val[j * C + i];
      square_sum +=
          (x_val[j * C + i] - means[i]) * (x_val[j * C + i] - means[i]);
    }
    batch_size[i] = 1;
    batch_sum[i] = val_sum / N;
    batch_square_sum[i] = square_sum / N + squared_sum_epsilon;
  }
}

template <typename T>
__global__ void KernelUpdateParam(int C,
                                  const T *d_batch_size,
                                  const T *d_batch_sum,
                                  const T *d_batch_square_sum,
                                  T *batch_size,
                                  T *batch_sum,
                                  T *batch_square_sum,
                                  const float decay_rate) {
  CUDA_KERNEL_LOOP(i, C) {
    batch_size[i] = batch_size[i] * decay_rate + d_batch_size[i];
    batch_sum[i] = batch_sum[i] * decay_rate + d_batch_sum[i];
    batch_square_sum[i] =
        batch_square_sum[i] * decay_rate + d_batch_square_sum[i];
  }
}

template <typename T, typename Context>
void DataNormGradKernel(const Context &dev_ctx,
                        const paddle::optional<DenseTensor> &scale_w_in,
                        const paddle::optional<DenseTensor> &bias_in,
                        const DenseTensor &x_in,
                        const DenseTensor &means_in,
                        const DenseTensor &scales_in,
                        const DenseTensor &out_grad,
                        float epsilon,
                        int slot_dim,
                        float summary_decay_rate,
                        bool enable_scale_and_shift,
                        const std::string &data_layout_in,
                        bool sync_stats,
                        DenseTensor *batch_size,
                        DenseTensor *batch_sum,
                        DenseTensor *batch_square_sum,
                        DenseTensor *scale_w_grad,
                        DenseTensor *bias_grad,
                        DenseTensor *x_grad,
                        DenseTensor *batch_size_grad,
                        DenseTensor *batch_sum_grad,
                        DenseTensor *batch_square_sum_grad) {
  const auto *x = &x_in;
  const auto *d_y = &out_grad;
  const auto *scales = &scales_in;
  const auto *means = &means_in;
  const float dr = summary_decay_rate;
  const bool need_sync_stats = sync_stats;

  const auto &x_dims = x->dims();
  // Align with CPU version, but should we add this restriction?
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      2,
      common::errors::PreconditionNotMet("The Input dim size should be 2"));
  const int N = x_dims[0];
  const int C = x_dims[1];

  // init output
  phi::DenseTensor *d_x = nullptr;
  if (x_grad != nullptr) {
    d_x = x_grad;
  }
  T *d_batch_size = dev_ctx.template Alloc<T>(batch_size_grad);
  T *d_batch_sum = dev_ctx.template Alloc<T>(batch_sum_grad);
  T *d_batch_square_sum = dev_ctx.template Alloc<T>(batch_square_sum_grad);

  auto stream = dev_ctx.stream();
  if (d_x != nullptr) {
    KernelDataNormBP<<<GET_BLOCKS(C * N), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        N,
        C,
        d_y->data<T>(),
        scales->data<T>(),
        dev_ctx.template Alloc<T>(d_x));
  }

  KernelDataNormBPStat<<<GET_BLOCKS(C), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
      N,
      C,
      x->data<T>(),
      means->data<T>(),
      epsilon,
      d_batch_size,
      d_batch_sum,
      d_batch_square_sum);

  if (need_sync_stats) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto comm_ctx = static_cast<phi::distributed::NCCLCommContext *>(
        dev_ctx.GetCommContext());
    PADDLE_ENFORCE_NE(comm_ctx,
                      nullptr,
                      common::errors::Unavailable(
                          "NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(
        reinterpret_cast<const void *>(d_batch_size),
        reinterpret_cast<void *>(d_batch_size),
        C,
        phi::ToNCCLDataType(x->dtype()),
        ncclSum,
        comm_ctx->GetNcclComm(),
        stream));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::ncclAllReduce(reinterpret_cast<const void *>(d_batch_sum),
                                    reinterpret_cast<void *>(d_batch_sum),
                                    C,
                                    phi::ToNCCLDataType(x->dtype()),
                                    ncclSum,
                                    comm_ctx->GetNcclComm(),
                                    stream));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclAllReduce(
        reinterpret_cast<const void *>(d_batch_square_sum),
        reinterpret_cast<void *>(d_batch_square_sum),
        C,
        phi::ToNCCLDataType(x->dtype()),
        ncclSum,
        comm_ctx->GetNcclComm(),
        stream));

    phi::backends::gpu::GpuStreamSync(stream);
#else
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU, and need_sync_stats connot be "
        "supported on windows now."));
#endif
  }

  T *batch_size_data = dev_ctx.template Alloc<T>(batch_size);
  T *batch_sum_data = dev_ctx.template Alloc<T>(batch_sum);
  T *batch_square_sum_data = dev_ctx.template Alloc<T>(batch_square_sum);
  KernelUpdateParam<<<GET_BLOCKS(C), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
      C,
      d_batch_size,
      d_batch_sum,
      d_batch_square_sum,
      batch_size_data,
      batch_sum_data,
      batch_square_sum_data,
      dr);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    data_norm_grad, GPU, ALL_LAYOUT, phi::DataNormGradKernel, float, double) {}
