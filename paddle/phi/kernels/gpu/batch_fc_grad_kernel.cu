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
__global__ void add_bias_grad_kernel(const T* dout_data,
                                     int slot_pairs_num,
                                     int ins_num,
                                     int out_dim,
                                     T* db_data) {
  CUDA_KERNEL_LOOP(idx, slot_pairs_num * out_dim) {
    int row = idx / out_dim;
    int col = idx % out_dim;
    T temp = static_cast<T>(0);
    for (int i = 0; i < ins_num; ++i) {
      int select_indx = ((row + 1) * i + 1) * col;
      temp += dout_data[select_indx];
    }
    db_data[idx] += temp;
  }
}

template <typename T>
void add_bias_grad(gpuStream_t stream,
                   const T* dout_data,
                   int slot_pairs_num,
                   int ins_num,
                   int out_dim,
                   T* db_data) {
  add_bias_grad_kernel<<<GET_BLOCKS(slot_pairs_num * out_dim),
                         CUDA_NUM_THREADS,
                         0,
                         stream>>>(
      dout_data, slot_pairs_num, ins_num, out_dim, db_data);
}

template <typename T, typename Context>
void BatchFCGradOpCUDAKernel(const Context& dev_ctx,
                             const DenseTensor& input_in,
                             const DenseTensor& w_in,
                             const DenseTensor& bias_in UNUSED,
                             const DenseTensor& out_grad,
                             DenseTensor* input_grad,
                             DenseTensor* w_grad,
                             DenseTensor* bias_grad) {
  auto* input = &input_in;
  auto* w = &w_in;
  auto* dout = &out_grad;

  auto* dx = input_grad;
  auto* dw = w_grad;
  auto* db = bias_grad;

  auto input_dims = input->dims();
  auto w_dims = w->dims();
  auto slot_pairs_num = input_dims[0];
  auto ins_num = input_dims[1];
  auto in_dim = input_dims[2];
  auto out_dim = w_dims[2];

  auto& place = *dev_ctx.eigen_device();
  // initialize
  dev_ctx.template Alloc<T>(dx);
  auto dx_eigen = phi::EigenVector<T>::Flatten(*dx);
  dx_eigen.device(place) = dx_eigen.constant(static_cast<T>(0));

  dev_ctx.template Alloc<T>(dw);
  auto dw_eigen = phi::EigenVector<T>::Flatten(*dw);
  dw_eigen.device(place) = dw_eigen.constant(static_cast<T>(0));

  // get data ptr
  const T* x_data = input->data<T>();
  const T* w_data = w->data<T>();
  const T* dout_data = dout->data<T>();
  T* dx_data = dx->data<T>();
  T* dw_data = dw->data<T>();

  dev_ctx.template Alloc<T>(db);
  auto db_eigen = phi::EigenVector<T>::Flatten(*db);
  db_eigen.device(place) = db_eigen.constant(static_cast<T>(0));
  T* db_data = db->data<T>();
  add_bias_grad<T>(
      dev_ctx.stream(), dout_data, slot_pairs_num, ins_num, out_dim, db_data);

  auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
  T alpha = 1;
  T beta = 0;

  // dx = dout_data * y^T
  blas.BatchedGEMM(CblasNoTrans,
                   CblasTrans,
                   ins_num,
                   in_dim,
                   out_dim,
                   alpha,
                   dout_data,
                   w_data,
                   beta,
                   dx_data,
                   slot_pairs_num,
                   ins_num * out_dim,
                   out_dim * in_dim);
  // dy = x^T * dout_data
  blas.BatchedGEMM(CblasTrans,
                   CblasNoTrans,
                   in_dim,
                   out_dim,
                   ins_num,
                   alpha,
                   x_data,
                   dout_data,
                   beta,
                   dw_data,
                   slot_pairs_num,
                   in_dim * ins_num,
                   ins_num * out_dim);
}
}  // namespace phi

PD_REGISTER_KERNEL(batch_fc_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchFCGradOpCUDAKernel,
                   float,
                   double) {}
