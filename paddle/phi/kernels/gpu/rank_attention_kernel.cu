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
void RankAttentionCUDAKernel(const Context &dev_ctx,
                             const DenseTensor &x,
                             const DenseTensor &rank_offset,
                             const DenseTensor &rank_param,
                             int max_rank,
                             int max_size,
                             DenseTensor *input_help,
                             DenseTensor *out,
                             DenseTensor *ins_rank) {
  // check dims
  auto x_dims = x.dims();
  auto ins_num = x_dims[0];
  auto x_fea_dim = x_dims[1];
  auto para_dims = rank_param.dims();
  auto para_row = para_dims[0];
  auto para_col = para_dims[1];
  auto rank_offset_dims = rank_offset.dims();
  auto *param = &rank_param;
  PADDLE_ENFORCE_EQ(
      rank_offset_dims[0],
      ins_num,
      common::errors::InvalidArgument("Input(RankOffset) has wrong rows."));
  PADDLE_ENFORCE_EQ(
      (rank_offset_dims[1] - 1) / 2,
      max_rank,
      common::errors::InvalidArgument("Input(RankOffset) has wrong columns."));
  PADDLE_ENFORCE_EQ(
      max_rank * max_rank * x_fea_dim,
      para_row,
      common::errors::InvalidArgument("Input(RankParam) has wrong rows."));

  int block_matrix_row = max_rank * x_fea_dim;
  int max_ins = std::max(ins_num, static_cast<int64_t>(max_size));

  phi::DenseTensor param_help;
  param_help.Resize({max_ins * block_matrix_row, para_col});
  dev_ctx.template Alloc<T>(&param_help);

  input_help->Resize({max_ins, block_matrix_row});
  ins_rank->Resize({max_ins, 1});
  dev_ctx.template Alloc<T>(input_help);
  dev_ctx.template Alloc<T>(ins_rank);
  dev_ctx.template Alloc<T>(out);

  // initialize
  auto param_help_eigen = phi::EigenVector<T>::Flatten(param_help);
  auto input_help_eigen = phi::EigenVector<T>::Flatten(*input_help);
  auto ins_rank_eigen = phi::EigenVector<T>::Flatten(*ins_rank);
  auto out_eigen = phi::EigenVector<T>::Flatten(*out);

  auto &place = *dev_ctx.eigen_device();

  param_help_eigen.device(place) = param_help_eigen.constant(static_cast<T>(0));
  input_help_eigen.device(place) = input_help_eigen.constant(static_cast<T>(0));
  ins_rank_eigen.device(place) = ins_rank_eigen.constant(static_cast<T>(-1));
  out_eigen.device(place) = out_eigen.constant(static_cast<T>(0));

  // get data ptr
  T *input_help_data = input_help->data<T>();
  T *param_help_data = param_help.data<T>();
  T *ins_rank_data = ins_rank->data<T>();
  T *out_data = out->data<T>();

  expand_rank_attention_input(dev_ctx.stream(),
                              x.data<T>(),
                              ins_num,
                              x_fea_dim,
                              input_help_data,
                              ins_num,
                              block_matrix_row,
                              rank_offset.data<int>(),
                              rank_offset_dims[0],
                              rank_offset_dims[1],
                              ins_rank_data,
                              max_rank);

  expand_rank_attention_param(dev_ctx.stream(),
                              x.data<T>(),
                              ins_num,
                              x_fea_dim,
                              rank_offset.data<int>(),
                              rank_offset_dims[0],
                              rank_offset_dims[1],
                              param->data<T>(),
                              para_row,
                              para_col,
                              param_help_data,
                              ins_num * block_matrix_row,
                              para_col,
                              max_rank);

  CBLAS_TRANSPOSE transA = CblasNoTrans;
  CBLAS_TRANSPOSE transB = CblasNoTrans;

  T alpha = 1;
  T beta = 0;
  int64_t strideA = block_matrix_row;
  int64_t strideB = block_matrix_row * para_col;

  auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
  blas.BatchedGEMM(transA,
                   transB,
                   1,
                   para_col,
                   block_matrix_row,
                   alpha,
                   input_help_data,
                   param_help_data,
                   beta,
                   out_data,
                   ins_num,
                   strideA,
                   strideB);
}

}  // namespace phi

PD_REGISTER_KERNEL(rank_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::RankAttentionCUDAKernel,
                   float,
                   double) {}
