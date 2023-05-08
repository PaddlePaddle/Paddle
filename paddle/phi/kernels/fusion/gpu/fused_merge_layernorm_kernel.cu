/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
   Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include "glog/logging.h"

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/fusion/gpu/fused_merge_layernorm.h"

namespace phi {
namespace fusion {

#define FINAL_MASK 0xffffffff

template <typename T>
__global__ void merge_layernorm_v2(T *out,
                                   const T *__restrict input,
                                   const T *__restrict gamma,
                                   const T *__restrict beta,
                                   const float layernorm_eps,
                                   int batch,
                                   int H,
                                   int W,
                                   int n) {
  // input is [batch, 2*H, 2*W, n/4]
  // output is [batch, H, W, n]
  // grid (W, H, batch)
  // block (n)
  const int kIte = 4;
  const int tid = threadIdx.x;
  const int W_idx = blockIdx.x;
  const int H_idx = blockIdx.y;
  const size_t batch_offset = blockIdx.z * H * W * n;
  const int input_H_stride = W * n / 2;
  const int output_H_stride = W * n;
  const int n_4 = n >> 2;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;
  float local_out[kIte];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kIte; i++) {
    int col_id = i * blockDim.x + tid;
    if (col_id < n) {
      int part_id = col_id / n_4;
      int offset_in_W = part_id / 2;
      int offset_in_H = part_id % 2;
      size_t input_id = batch_offset +
                        (2 * H_idx + offset_in_H) * input_H_stride +
                        (2 * W_idx + offset_in_W) * n_4 + (col_id % n_4);
      local_out[i] = static_cast<float>(__ldg(input + input_id));
      sum += local_out[i];
    }
  }

  mean = phi::funcs::BlockReduceSum<float>(sum, FINAL_MASK);
  if (tid == 0) {
    s_mean = mean / n;
  }
  __syncthreads();

  float var = 0.0f;
#pragma unroll
  for (int i = 0; i < kIte; i++) {
    int col_id = i * blockDim.x + tid;
    if (col_id < n) {
      local_out[i] = local_out[i] - s_mean;
      var += local_out[i] * local_out[i];
    }
  }

  variance = phi::funcs::BlockReduceSum<float>(var, FINAL_MASK);
  if (tid == 0) {
    s_variance = rsqrtf(variance / n + layernorm_eps);
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < kIte; i++) {
    int col_id = i * blockDim.x + tid;
    if (col_id < n) {
      size_t output_idx =
          batch_offset + H_idx * output_H_stride + W_idx * n + col_id;
      out[output_idx] =
          static_cast<T>(local_out[i] * s_variance *
                             static_cast<float>(__ldg(&gamma[col_id])) +
                         static_cast<float>(__ldg(&beta[col_id])));
    }
  }
}

template <typename T>
void invokeMergeLayernorm(T *output,
                          const T *input,
                          const T *gamma,
                          const T *beta,
                          float layernorm_eps,
                          int batch,
                          int H,
                          int W,
                          int n,
                          cudaStream_t stream) {
  if ((W % 2 != 0) || (H % 2 != 0)) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "H(W) of merge layernorm should be a multiple of 2."));
  }
  dim3 grid(W / 2, H / 2, batch);
  int blockSize = (n + 31) / 32 * 32;
  merge_layernorm_v2<T><<<grid, blockSize, 0, stream>>>(
      output, input, gamma, beta, layernorm_eps, batch, H / 2, W / 2, n * 4);
}

template void invokeMergeLayernorm<float>(float *output,
                                          const float *input,
                                          const float *gamma,
                                          const float *beta,
                                          float layernorm_eps,
                                          int batch,
                                          int H,
                                          int W,
                                          int n,
                                          cudaStream_t stream);

template void invokeMergeLayernorm<half>(half *output,
                                         const half *input,
                                         const half *gamma,
                                         const half *beta,
                                         float layernorm_eps,
                                         int batch,
                                         int H,
                                         int W,
                                         int n,
                                         cudaStream_t stream);

template <typename T, typename Context>
void FusedMergeLayernormKernel(const Context &dev_ctx,
                               const DenseTensor &x,
                               const DenseTensor &scale,
                               const DenseTensor &bias,
                               const float epsilon,
                               const int begin_norm_axis,
                               DenseTensor *out) {
  auto *x_data = x.data<T>();
  auto *bias_data = bias.data<T>();
  auto *scale_data = scale.data<T>();
  auto *y_data = dev_ctx.template Alloc<T>(out);

  auto x_dim = x.dims();
  int batch = x_dim[0];
  int input_resolution = static_cast<int>(std::sqrt(x_dim[1]));
  int dim = static_cast<int>(x_dim[2]);
  PADDLE_ENFORCE_EQ(
      input_resolution * input_resolution,
      x_dim[1],
      phi::errors::InvalidArgument(
          "The MergeLayernorm TRT Plugin get invalid input_resolution %d",
          input_resolution));

  auto stream = dev_ctx.stream();
  if (std::is_same<T, phi::dtype::float16>::value) {
    VLOG(3) << "TRT Plugin DataType selected. MergeLayernorm-->fp16";
    invokeMergeLayernorm<half>(reinterpret_cast<half *>(y_data),
                               reinterpret_cast<const half *>(x_data),
                               reinterpret_cast<const half *>(scale_data),
                               reinterpret_cast<const half *>(bias_data),
                               epsilon,
                               batch,
                               input_resolution,
                               input_resolution,
                               dim,
                               stream);
  } else if (std::is_same<T, float>::value) {
    VLOG(3) << "TRT Plugin DataType selected. MergeLayernorm-->fp32";
    invokeMergeLayernorm<float>(reinterpret_cast<float *>(y_data),
                                reinterpret_cast<const float *>(x_data),
                                reinterpret_cast<const float *>(scale_data),
                                reinterpret_cast<const float *>(bias_data),
                                epsilon,
                                batch,
                                input_resolution,
                                input_resolution,
                                dim,
                                stream);
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "The MergeLayernorm TRT Plugin's input type should be float or half."));
  }
}

}  // namespace fusion
}  // namespace phi

#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 10000
PD_REGISTER_KERNEL(fused_merge_layernorm,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedMergeLayernormKernel,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(fused_merge_layernorm,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedMergeLayernormKernel,
                   float) {}
#endif
