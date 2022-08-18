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

#include "paddle/phi/kernels/impl/spectral_norm_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void SpectralNormGradKernel(const Context& dev_ctx,
                            const DenseTensor& weight,
                            const DenseTensor& u,
                            const DenseTensor& v,
                            const DenseTensor& out_grad,
                            int dim,
                            int power_iters,
                            float eps,
                            DenseTensor* weight_grad) {
  auto& place = *dev_ctx.eigen_device();
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  const int h = u.dims()[0];
  const int w = v.dims()[0];

  DenseTensor weight_mat, out_grad_mat;
  auto dims = weight.dims();
  const int rank = dims.size();
  std::vector<int> real_dims;
  if (dim != 0) {
    std::vector<int> perm;
    perm.push_back(dim);
    real_dims.push_back(dims[dim]);
    for (int i = 0; i < rank; i++) {
      if (i != dim) {
        perm.push_back(i);
        real_dims.push_back(dims[i]);
      }
    }
    weight_mat.Resize(phi::make_ddim(real_dims));
    dev_ctx.template Alloc<T>(&weight_mat);
    out_grad_mat.Resize(phi::make_ddim(real_dims));
    dev_ctx.template Alloc<T>(&out_grad_mat);
    TransCompute2DTo5D<Context, T>(dev_ctx, weight, rank, perm, &weight_mat);
    TransCompute2DTo5D<Context, T>(
        dev_ctx, out_grad, rank, perm, &out_grad_mat);
  } else {
    for (int i = 0; i < rank; i++) {
      real_dims.push_back(i);
    }
    phi::Copy(dev_ctx, weight, dev_ctx.GetPlace(), true, &weight_mat);
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), true, &out_grad_mat);
  }
  weight_mat = weight_mat.Resize({h, w});
  out_grad_mat = out_grad_mat.Resize({h, w});

  DenseTensor sigma;
  sigma.Resize(weight_mat.dims());
  dev_ctx.template Alloc<T>(&sigma);
  DenseTensor uu, vv;
  phi::Copy(dev_ctx, u, dev_ctx.GetPlace(), true, &uu);
  phi::Copy(dev_ctx, v, dev_ctx.GetPlace(), true, &vv);
  CalcMatrixSigmaAndNormWeight<Context, T>(dev_ctx,
                                           &weight_mat,
                                           &(uu.Resize({h, 1})),
                                           &(vv.Resize({w, 1})),
                                           &sigma,
                                           power_iters,
                                           eps);

  DenseTensor uv;
  uv.Resize({h, w});
  dev_ctx.template Alloc<T>(&uv);
  blas.MatMul(
      uu.Resize({h, 1}), false, vv.Resize({w, 1}), false, T(1), &uv, T(0));

  DenseTensor weight_grad_mat;
  weight_grad_mat.Resize({h, w});
  dev_ctx.template Alloc<T>(&weight_grad_mat);
  auto weight_grad_mat_t = EigenTensor<T, 2>::From(weight_grad_mat);
  auto weight_mat_t = EigenTensor<T, 2>::From(weight_mat);
  auto out_grad_mat_t = EigenTensor<T, 2>::From(out_grad_mat);
  auto sigma_t = EigenTensor<T, 2>::From(sigma);
  auto uv_t = EigenTensor<T, 2>::From(uv);
  weight_mat_t.device(place) =
      weight_mat_t.sum().eval().reshape(Array2(1, 1)).broadcast(Array2(h, w));
  weight_grad_mat_t.device(place) =
      out_grad_mat_t * (out_grad_mat_t.constant(1.0) - uv_t * weight_mat_t) /
      sigma_t;

  if (dim != 0) {
    std::vector<int> perm;
    for (int i = 0; i < rank; i++) {
      if (i < dim) {
        perm.push_back(i + 1);
      } else if (i == dim) {
        perm.push_back(0);
      } else {
        perm.push_back(i);
      }
    }
    weight_grad->Resize(dims);
    dev_ctx.template Alloc<T>(weight_grad);
    TransCompute2DTo5D<Context, T>(
        dev_ctx,
        weight_grad_mat.Resize(phi::make_ddim(real_dims)),
        rank,
        perm,
        weight_grad);
  } else {
    phi::Copy(dev_ctx,
              weight_grad_mat.Resize(dims),
              dev_ctx.GetPlace(),
              true,
              weight_grad);
  }
}

}  // namespace phi
