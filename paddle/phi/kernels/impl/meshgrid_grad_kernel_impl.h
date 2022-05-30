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

#include "paddle/phi/kernels/meshgrid_grad_kernel.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename T, typename Context, int Rank>
void MeshgridBackward(const Context& ctx,
                      const std::vector<const DenseTensor*>& ins,
                      const std::vector<const DenseTensor*>& out_grad,
                      std::vector<DenseTensor*> outs) {
  int n = out_grad.size();
  auto out_dims = out_grad[0]->dims();

  for (int i = 0; i < n; i++) {
    ctx.template Alloc<T>(outs[i]);
    auto out_grad_tmp = EigenVector<T>::Flatten(*out_grad[i]);
    auto in_grad = EigenVector<T>::Flatten(*outs[i]);

    std::vector<int> reduce_dims_vec;
    std::vector<int> reshape_dims_vec;
    for (int j = 0; j < n; j++) {
      reduce_dims_vec.push_back(reshape_dims_vec.size());
      if (j == i) {
        reshape_dims_vec.push_back(1);
        reshape_dims_vec.push_back(out_dims[j]);
      } else {
        reshape_dims_vec.push_back(out_dims[j]);
        reshape_dims_vec.push_back(1);
      }
    }

    Eigen::DSizes<Eigen::DenseIndex, Rank> reduce_dims;
    for (int k = 0; k < n; k++) {
      reduce_dims[k] = reduce_dims_vec[k];
    }

    Eigen::DSizes<Eigen::DenseIndex, Rank * 2> reshape_dims;
    for (int k = 0; k < n * 2; k++) {
      reshape_dims[k] = reshape_dims_vec[k];
    }

    auto& place = *ctx.eigen_device();
    funcs::EigenBroadcastGrad<std::decay_t<decltype(place)>, T, Rank>::Eval(
        place, in_grad, out_grad_tmp, reduce_dims, reshape_dims);
  }
}

template <typename T, typename Context>
void MeshgridGradKernel(const Context& ctx,
                        const std::vector<const DenseTensor*>& inputs,
                        const std::vector<const DenseTensor*>& outputs_grad,
                        std::vector<DenseTensor*> inputs_grad) {
  int n = outputs_grad.size();
  switch (n) {
    case 1:
      MeshgridBackward<T, Context, 1>(ctx, inputs, outputs_grad, inputs_grad);
      break;
    case 2:
      MeshgridBackward<T, Context, 2>(ctx, inputs, outputs_grad, inputs_grad);
      break;
    case 3:
      MeshgridBackward<T, Context, 3>(ctx, inputs, outputs_grad, inputs_grad);
      break;
    case 4:
      MeshgridBackward<T, Context, 4>(ctx, inputs, outputs_grad, inputs_grad);
      break;
    case 5:
      MeshgridBackward<T, Context, 5>(ctx, inputs, outputs_grad, inputs_grad);
      break;
    case 6:
      MeshgridBackward<T, Context, 6>(ctx, inputs, outputs_grad, inputs_grad);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Excepted Tensor numbers between 1 and 6, but only received d% .",
          n));
  }
}

}  // namespace phi
