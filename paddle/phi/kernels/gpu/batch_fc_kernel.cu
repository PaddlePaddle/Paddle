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

#include <string>

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

const int CUDA_NUM_THREADS = 1024;
static inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void add_bias_kernel(
    T* data, int slot_pairs_num, int ins_num, int out_dim, const T* bias) {
  CUDA_KERNEL_LOOP(idx, slot_pairs_num * ins_num * out_dim) {
    int block_len = ins_num * out_dim;
    int slot_index = idx / block_len;
    int out_dim_index = (idx % block_len) % out_dim;
    T temp = data[idx] + bias[slot_index * out_dim + out_dim_index];
    data[idx] = temp;
  }
}

template <typename T>
void add_bias(gpuStream_t stream,
              T* data,
              int slot_pairs_num,
              int ins_num,
              int out_dim,
              const T* bias) {
  add_bias_kernel<<<GET_BLOCKS(slot_pairs_num * ins_num * out_dim),
                    CUDA_NUM_THREADS,
                    0,
                    stream>>>(data, slot_pairs_num, ins_num, out_dim, bias);
}

template <typename T, typename Context>
void BatchFCCUDAKernel(const Context& dev_ctx,
                       const DenseTensor& input_in,
                       const DenseTensor& w_in,
                       const DenseTensor& bias_in,
                       DenseTensor* out) {
  // X.dim = slot_pairs_num * ins_num * in_dim
  // W.dim = slot_pairs_num * in_dim * out_dim
  // b.dim = slot_pairs_num * out_dim
  // output.dim = slot_pairs_num * ins_num * out_dim
  auto* input = &input_in;
  auto* w = &w_in;
  auto* bias = &bias_in;
  auto* output = out;
  auto input_dims = input->dims();
  auto w_dims = w->dims();
  auto slot_pairs_num = input_dims[0];
  auto ins_num = input_dims[1];
  auto in_dim = input_dims[2];
  auto out_dim = w_dims[2];

  // get data ptr
  const T* in_data = input->data<T>();
  const T* w_data = w->data<T>();
  const T* bias_data = bias->data<T>();

  output->Resize({slot_pairs_num, ins_num, out_dim});
  T* out_data = dev_ctx.template Alloc<T>(output);
  // initialize
  auto out_eigen = phi::EigenVector<T>::Flatten(*output);
  auto& place = *dev_ctx.eigen_device();
  out_eigen.device(place) = out_eigen.constant(static_cast<T>(0));

  CBLAS_TRANSPOSE transA = CblasNoTrans;
  CBLAS_TRANSPOSE transB = CblasNoTrans;

  T alpha = 1;
  T beta = 0;
  int64_t strideA = ins_num * in_dim;
  int64_t strideB = in_dim * out_dim;

  auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
  blas.BatchedGEMM(transA,
                   transB,
                   ins_num,
                   out_dim,
                   in_dim,
                   alpha,
                   in_data,
                   w_data,
                   beta,
                   out_data,
                   slot_pairs_num,
                   strideA,
                   strideB);
  add_bias<T>(
      dev_ctx.stream(), out_data, slot_pairs_num, ins_num, out_dim, bias_data);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    batch_fc, GPU, ALL_LAYOUT, phi::BatchFCCUDAKernel, float, double) {}
