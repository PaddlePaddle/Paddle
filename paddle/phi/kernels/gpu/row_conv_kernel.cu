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

#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/phi/kernels/funcs/math_function.h"
namespace phi {

namespace {

static inline int DivUp(int x, int y) { return (x + y - 1) / y; }

// Forward prop (shared memory version, for small future_context)
template <typename T>
__global__ void RowConvForwardSharedMemory(const T *in,
                                           const T *wt,
                                           int num_sequence,
                                           int input_dim,
                                           int future_context,
                                           const size_t *batch_indices,
                                           T *out) {
  int blx = blockDim.x;
  int bly = blockDim.y;
  int thx = threadIdx.x;
  int thy = threadIdx.y;
  int d = blockIdx.x * blx + thx;  // index along input dim

  extern __shared__ T mem[];
  T *sw = mem;

  if (thy < future_context) {
    sw[thy * blx + thx] =
        (d < input_dim) ? wt[thy * input_dim + d] : static_cast<T>(0);
  }
  __syncthreads();
  for (size_t i = 0; i < num_sequence; i++) {
    int start = static_cast<int>(batch_indices[i]);
    int end = static_cast<int>(batch_indices[i + 1]);
    int current_timesteps = end - start;

    for (int k = thy; k < current_timesteps; k += bly) {
      T sum = 0;
      for (int w = 0; (w < future_context) && ((k + w) < current_timesteps);
           w++) {
        sum += (d < input_dim)
                   ? sw[w * blx + thx] * in[(start + k + w) * input_dim + d]
                   : static_cast<T>(0);
      }
      if (d < input_dim) {
        out[(start + k) * input_dim + d] = sum;
      }
    }
  }
}

// Forward prop (naive version)
template <typename T>
__global__ void RowConvForward(const T *in,
                               const T *wt,
                               int num_sequence,
                               int input_dim,
                               int future_context,
                               const size_t *batch_indices,
                               T *out) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;  // index along input_dim
  int bly = blockDim.y;
  int thy = threadIdx.y;

  if (d >= input_dim) return;
  for (size_t i = 0; i < num_sequence; i++) {
    int start = static_cast<int>(batch_indices[i]);
    int end = static_cast<int>(batch_indices[i + 1]);
    int current_timesteps = end - start;

    for (int k = thy; k < current_timesteps; k += bly) {
      T sum = 0;
      for (int w = 0; (w < future_context) && ((k + w) < current_timesteps);
           w++) {
        sum += (wt[w * input_dim + d] * in[(start + k + w) * input_dim + d]);
      }
      out[(start + k) * input_dim + d] = sum;
    }
  }
}
}  // namespace

template <typename T, typename Context>
void RowConvKernel(const Context &dev_ctx,
                   const DenseTensor &x_in,
                   const DenseTensor &filter_in,
                   DenseTensor *Out) {
  auto *X = &x_in;
  auto *Filter = &filter_in;

  const T *in = X->data<T>();
  const T *weight = Filter->data<T>();
  T *out = dev_ctx.template Alloc<T>(Out);
  bool is_tensor = X->lod().empty();
  int batch_size = 0;
  if (is_tensor) {
    batch_size = X->dims()[0];
  } else {
    batch_size = X->lod()[0].size() - 1;
  }
  int input_dim = 0;
  phi::Vector<size_t> batch_indices(batch_size + 1);
  int timesteps = X->dims()[1];
  if (is_tensor) {
    for (int i = 0; i < batch_size + 1; i++) {
      batch_indices[i] = i * timesteps;
    }
    input_dim = X->dims()[2];
  } else {
    batch_indices = X->lod()[0];
    input_dim = X->dims()[1];
  }

  int num_sequence = batch_indices.size() - 1;
  int future_context = Filter->dims()[0];
  phi::MixVector<size_t> mix_vector(&batch_indices);
  size_t *idx = mix_vector.CUDAMutableData(dev_ctx.GetPlace());
  auto stream = dev_ctx.stream();

  if (future_context <= 32) {
    dim3 block_dim = dim3(32, 32);
    dim3 grid_dim = dim3(DivUp(input_dim, block_dim.x), 1);
    int mem_per_block = (future_context * block_dim.x) * sizeof(T);
    RowConvForwardSharedMemory<T>
        <<<grid_dim, block_dim, mem_per_block, stream>>>(
            in, weight, num_sequence, input_dim, future_context, idx, out);
  } else {
    dim3 block_dim = dim3(32, 32);
    dim3 grid_dim = dim3(DivUp(input_dim, block_dim.x), 1);
    RowConvForward<T><<<grid_dim, block_dim, 0, stream>>>(
        in, weight, num_sequence, input_dim, future_context, idx, out);
  }
  mix_vector.CopyToCPU();
}
}  // namespace phi

PD_REGISTER_KERNEL(row_conv, GPU, ALL_LAYOUT, phi::RowConvKernel, float) {}
