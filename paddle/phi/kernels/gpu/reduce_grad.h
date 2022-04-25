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

#pragma once

// CUDA and HIP use same api
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"

namespace phi {

template <typename InT, typename Functor>
void ReduceGrad(const GPUContext& dev_ctx,
                DenseTensor* d_out,
                DenseTensor* d_x,
                DataType out_dtype,
                Functor functor) {
  std::vector<const DenseTensor*> inputs = {d_out};
  std::vector<DenseTensor*> outputs = {d_x};
  PD_VISIT_ALL_TYPES(
      out_dtype, "BroadcastKernel", ([&] {
        funcs::BroadcastKernel<phi::ElementwiseType::kUnary, InT, data_t>(
            dev_ctx, inputs, &outputs, 0, functor);
      }));
}

template <typename T,
          typename Context,
          template <typename, typename> class TransformOp>
void ReduceGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      const std::vector<int64_t>& dims,
                      bool keep_dim,
                      bool reduce_all,
                      DenseTensor* x_grad) {
  auto* in_x = &x;
  auto* d_out = &out_grad;
  auto* d_x = x_grad;

  auto pt_out_dtype = x.dtype();

  // get reduce_dim and reduce_num for reduce_mean_grad
  int dim_size = in_x->dims().size();
  std::vector<int> reduce_dims =
      funcs::details::GetReduceDim(dims, dim_size, reduce_all);

  auto update_dims = vectorize(d_x->dims());
  int reduce_num = 1;
  for (auto i : reduce_dims) {
    reduce_num *= (in_x->dims())[i];
    update_dims[i] = 1;
  }
  // make new tensor
  DenseTensor new_d_out(d_out->dtype());
  new_d_out.ShareDataWith(*d_out);
  new_d_out.Resize(phi::make_ddim(update_dims));

  dev_ctx.Alloc(d_x, x.dtype());

  auto pt_d_out = new_d_out;
  auto pt_d_x = *d_x;
  using MPType = typename kps::details::MPTypeTrait<T>::Type;

  phi::ReduceGrad<T, TransformOp<T, MPType>>(
      dev_ctx,
      &pt_d_out,
      &pt_d_x,
      pt_out_dtype,
      TransformOp<T, MPType>(reduce_num));
}

}  // namespace phi
#endif
