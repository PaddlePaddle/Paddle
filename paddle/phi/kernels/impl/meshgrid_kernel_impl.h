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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/meshgrid_kernel.h"

namespace phi {

template <typename T, typename Context, int Rank>
void MeshgridForward(const Context& ctx,
                     const std::vector<const DenseTensor*>& ins,
                     std::vector<DenseTensor*> outs) {
  PADDLE_ENFORCE_EQ(
      ins.size() > 1,
      true,
      phi::errors::InvalidArgument(
          "Expected at least 2 input tensors, but only received d%.",
          ins.size()));

  int64_t size = ins.size();
  std::vector<int64_t> shape(size);

  for (int64_t i = 0; i < size; i++) {
    switch (ins[i]->dims().size()) {
      case 0:
        shape[i] = 1;
        break;
      case 1:
        shape[i] = ins[i]->dims()[0];
        break;
      default:
        PADDLE_THROW(phi::errors::InvalidArgument(
            "Expected scalar or 1D tensor in the tensor list but got tensor "
            "%d: ",
            i));
    }
  }

  for (int64_t i = 0; i < size; i++) {
    std::vector<int64_t> view_shape(size, 1);
    view_shape[i] = shape[i];

    DenseTensor reshape_ins_tensor;
    phi::Copy(ctx, *ins[i], ctx.GetPlace(), false, &reshape_ins_tensor);
    DDim out_dims_reshape = common::make_ddim(view_shape);
    reshape_ins_tensor.Resize(out_dims_reshape);
    DDim out_dims = common::make_ddim(shape);

    Eigen::DSizes<Eigen::DenseIndex, Rank> bcast_dims;
    for (int64_t j = 0; j < size; j++) {
      bcast_dims[j] = shape[j];
    }
    bcast_dims[i] = 1;

    outs[i]->Resize(out_dims);
    auto x = EigenTensor<T, Rank>::From(
        static_cast<const DenseTensor>(reshape_ins_tensor));
    ctx.template Alloc<T>(outs[i]);
    auto y = EigenTensor<T, Rank>::From(*outs[i]);
    auto& place = *ctx.eigen_device();
    funcs::EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(
        place, y, x, bcast_dims);
  }
}

template <typename T, typename Context>
void MeshgridKernel(const Context& ctx,
                    const std::vector<const DenseTensor*>& inputs,
                    std::vector<DenseTensor*> outputs) {
  int rank = inputs.size();
  switch (rank) {
    case 1:
      MeshgridForward<T, Context, 1>(ctx, inputs, outputs);
      break;
    case 2:
      MeshgridForward<T, Context, 2>(ctx, inputs, outputs);
      break;
    case 3:
      MeshgridForward<T, Context, 3>(ctx, inputs, outputs);
      break;
    case 4:
      MeshgridForward<T, Context, 4>(ctx, inputs, outputs);
      break;
    case 5:
      MeshgridForward<T, Context, 5>(ctx, inputs, outputs);
      break;
    case 6:
      MeshgridForward<T, Context, 6>(ctx, inputs, outputs);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Excepted Tensor numbers between 1 and 6, but only received d% .",
          rank));
  }
}
}  // namespace phi
