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

#include <algorithm>

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/rank_attention.cu.h"

namespace phi {

template <typename T, typename Context>
void RankAttentionGradOpCUDAKernel(const Context &dev_ctx,
                                   const DenseTensor &x,
                                   const DenseTensor &rank_offset,
                                   const DenseTensor &rank_param,
                                   const DenseTensor &input_help,
                                   const DenseTensor &ins_rank,
                                   const DenseTensor &out_grad,
                                   int max_rank UNUSED,
                                   int max_size,
                                   DenseTensor *rank_param_grad) {
  auto *dout = &out_grad;
  auto *drank_para = rank_param_grad;

  // get dim
  auto x_dims = x.dims();
  auto ins_num = x_dims[0];
  auto x_fea_dim = x_dims[1];
  auto para_dims = rank_param.dims();
  auto para_row = para_dims[0];
  auto para_col = para_dims[1];
  auto rank_offset_dims = rank_offset.dims();
  auto rank_offset_max_rank =
      (rank_offset_dims[1] - 1) / 2;  // Not use param max_rank
  int block_matrix_row = rank_offset_max_rank * x_fea_dim;
  auto &place = *dev_ctx.eigen_device();

  int max_ins = std::max(ins_num, static_cast<int64_t>(max_size));
  // initialize out grad
  dev_ctx.template Alloc<T>(drank_para);
  auto drank_para_eigen = phi::EigenVector<T>::Flatten(*drank_para);
  drank_para_eigen.device(place) = drank_para_eigen.constant(static_cast<T>(0));

  // copy data
  phi::DenseTensor param_grad;
  param_grad.Resize({max_ins * block_matrix_row, para_col});
  dev_ctx.template Alloc<T>(&param_grad);

  // initialize
  auto param_grad_eigen = phi::EigenVector<T>::Flatten(param_grad);
  param_grad_eigen.device(place) = param_grad_eigen.constant(static_cast<T>(0));
  // get data ptr
  const T *input_help_data = input_help.data<T>();
  const T *ins_rank_data = ins_rank.data<T>();
  T *param_grad_data = param_grad.data<T>();

  auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
  T alpha = 1;
  T beta = 0;

  // get param_grad
  CBLAS_TRANSPOSE transA = CblasTrans;
  CBLAS_TRANSPOSE transB = CblasNoTrans;
  int64_t strideA = block_matrix_row;
  int64_t strideB = para_col;
  blas.BatchedGEMM(transA,
                   transB,
                   block_matrix_row,
                   para_col,
                   1,
                   alpha,
                   input_help_data,
                   dout->data<T>(),
                   beta,
                   param_grad_data,
                   ins_num,
                   strideA,
                   strideB);
  // merge param_grad to get drank_para
  merge_rank_attention_param_grad(dev_ctx.stream(),
                                  param_grad_data,
                                  ins_num * block_matrix_row,
                                  para_col,
                                  drank_para->data<T>(),
                                  para_row,
                                  para_col,
                                  ins_rank_data,
                                  ins_num,
                                  rank_offset_max_rank,
                                  x_fea_dim);
}
}  // namespace phi

PD_REGISTER_KERNEL(rank_attention_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RankAttentionGradOpCUDAKernel,
                   float,
                   double) {}
