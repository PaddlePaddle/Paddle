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

namespace phi {

template <typename T,
          size_t D,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using Tensor = framework::Tensor;

using Array1 = Eigen::DSizes<int64_t, 1>;
using Array2 = Eigen::DSizes<int64_t, 2>;
using IndexPair = Eigen::IndexPair<int>;

template <typename DeviceContext, typename T>
static inline void TransCompute(const int rank,
                                const Tensor& in,
                                Tensor* out,
                                const std::vector<int>& perm,
                                const DeviceContext& dev_ctx) {
  if (rank <= 1 || rank > 5) {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Weight rank of SpectralNorm should be in range [2, 5], but got %d.",
        rank));
  }

  switch (rank) {
    case 2:
      phi::funcs::Transpose<DeviceContext, T, 2> trans2;
      trans2(dev_ctx, in, out, perm);
      break;
    case 3:
      phi::funcs::Transpose<DeviceContext, T, 3> trans3;
      trans3(dev_ctx, in, out, perm);
      break;
    case 4:
      phi::funcs::Transpose<DeviceContext, T, 4> trans4;
      trans4(dev_ctx, in, out, perm);
      break;
    case 5:
      phi::funcs::Transpose<DeviceContext, T, 5> trans5;
      trans5(dev_ctx, in, out, perm);
      break;
    default:
      break;
  }
}

template <typename DeviceContext, typename T>
static inline void CalcMatrixSigmaAndNormWeight(
    Tensor* sigma,
    Tensor* u,
    Tensor* v,
    Tensor* weight,
    const int power_iters,
    const float eps,
    const framework::ExecutionContext& ctx) {
  auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
  auto blas = phi::funcs::GetBlas<DeviceContext, T>(ctx);
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
  Tensor weight_v;
  weight_v.mutable_data<T>({h, 1}, ctx.GetPlace());
  blas.MatMul(*weight, false, *v, false, T(1), &weight_v, T(0));
  auto weight_v_t = EigenTensor<T, 2>::From(weight_v);
  sigma_t.device(place) = (u_t * weight_v_t)
                              .sum()
                              .eval()
                              .reshape(Array2(1, 1))
                              .broadcast(Array2(h, w));
  weight_t.device(place) = weight_t / sigma_t;
}

}  // namespace phi
