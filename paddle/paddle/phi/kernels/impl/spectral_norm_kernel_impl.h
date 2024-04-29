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

#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

using Array1 = Eigen::DSizes<int64_t, 1>;
using Array2 = Eigen::DSizes<int64_t, 2>;
using IndexPair = Eigen::IndexPair<int>;

template <typename Context, typename T>
static inline void TransCompute2DTo5D(const Context& dev_ctx,
                                      const DenseTensor& in,
                                      const int rank,
                                      const std::vector<int>& perm,
                                      DenseTensor* out) {
  if (rank <= 1 || rank > 5) {
    PADDLE_THROW(phi::errors::Fatal(
        "Weight rank of SpectralNorm should be in range [2, 5], but got %d.",
        rank));
  }

  switch (rank) {
    case 2:
      phi::funcs::Transpose<Context, T, 2> trans2;
      trans2(dev_ctx, in, out, perm);
      break;
    case 3:
      phi::funcs::Transpose<Context, T, 3> trans3;
      trans3(dev_ctx, in, out, perm);
      break;
    case 4:
      phi::funcs::Transpose<Context, T, 4> trans4;
      trans4(dev_ctx, in, out, perm);
      break;
    case 5:
      phi::funcs::Transpose<Context, T, 5> trans5;
      trans5(dev_ctx, in, out, perm);
      break;
    default:
      break;
  }
}

template <typename Context, typename T>
static inline void CalcMatrixSigmaAndNormWeight(const Context& dev_ctx,
                                                DenseTensor* weight,
                                                DenseTensor* u,
                                                DenseTensor* v,
                                                DenseTensor* sigma,
                                                const int power_iters,
                                                const float eps) {
  auto& place = *dev_ctx.eigen_device();
  auto blas = funcs::GetBlas<Context, T>(dev_ctx);
  auto sigma_t = EigenTensor<T, 2>::From(*sigma);
  auto weight_t = EigenTensor<T, 2>::From(*weight);
  auto u_t = EigenTensor<T, 2>::From(*u);
  auto v_t = EigenTensor<T, 2>::From(*v);

  const int h = weight->dims()[0];
  const int w = weight->dims()[1];

  for (int i = 0; i < power_iters; i++) {
    // V = W^T * U / ||W^T * U||_2
    blas.MatMul(*weight, true, *u, false, T(1), v, T(0));
    auto v_t_norm =
        v_t.square().sum().sqrt().eval().reshape(Array1(1)).broadcast(
            Array1(w));
    v_t.device(place) = v_t / (v_t_norm + v_t_norm.constant(eps));
    // U = W^T * V / ||W^T * V||_2
    blas.MatMul(*weight, false, *v, false, T(1), u, T(0));
    auto u_t_norm =
        u_t.square().sum().sqrt().eval().reshape(Array1(1)).broadcast(
            Array1(h));
    u_t.device(place) = u_t / (u_t_norm + u_t_norm.constant(eps));
  }
  DenseTensor weight_v;
  weight_v.Resize({h, 1});
  dev_ctx.template Alloc<T>(&weight_v);
  blas.MatMul(*weight, false, *v, false, T(1), &weight_v, T(0));
  auto weight_v_t = EigenTensor<T, 2>::From(weight_v);
  sigma_t.device(place) = (u_t * weight_v_t)
                              .sum()
                              .eval()
                              .reshape(Array2(1, 1))
                              .broadcast(Array2(h, w));
  weight_t.device(place) = weight_t / sigma_t;
}

template <typename T, typename Context>
void SpectralNormKernel(const Context& dev_ctx,
                        const DenseTensor& weight,
                        const DenseTensor& u,
                        const DenseTensor& v,
                        int dim,
                        int power_iters,
                        float eps,
                        DenseTensor* out) {
  const int h = u.dims()[0];
  const int w = v.dims()[0];

  DenseTensor weight_mat;
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
    weight_mat.Resize(common::make_ddim(real_dims));
    dev_ctx.template Alloc<T>(&weight_mat);
    TransCompute2DTo5D<Context, T>(dev_ctx, weight, rank, perm, &weight_mat);
  } else {
    for (int i = 0; i < rank; i++) {
      real_dims.push_back(i);
    }
    phi::Copy(dev_ctx, weight, dev_ctx.GetPlace(), true, &weight_mat);
  }
  weight_mat = weight_mat.Resize({h, w});

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
    out->Resize(dims);
    dev_ctx.template Alloc<T>(out);
    TransCompute2DTo5D<Context, T>(
        dev_ctx,
        weight_mat.Resize(common::make_ddim(real_dims)),
        rank,
        perm,
        out);
  } else {
    phi::Copy(dev_ctx, weight_mat.Resize(dims), dev_ctx.GetPlace(), true, out);
  }
}

}  // namespace phi
