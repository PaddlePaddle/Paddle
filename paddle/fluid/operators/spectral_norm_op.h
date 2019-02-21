/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using Tensor = framework::Tensor;

using Array1 = Eigen::DSizes<int64_t, 1>;
using Array2 = Eigen::DSizes<int64_t, 2>;
using IndexPair = Eigen::IndexPair<int>;

static inline void ResizeWeight(Tensor* weight_mat, const int dim) {
  auto weight_dims = weight_mat->dims();
  int h = 1;
  int w = 1;
  for (int i = 0; i < weight_dims.size(); i++) {
    if (i <= dim) {
      h *= weight_dims[i];
    } else {
      w *= weight_dims[i];
    }
  }
  *weight_mat = weight_mat->Resize({h, w});
}

template <typename DeviceContext, typename T>
static inline void CalcMatrixSigmaAndNormWeight(
    Tensor* sigma, Tensor* u, Tensor* v, Tensor* weight, const int power_iters,
    const float eps, const framework::ExecutionContext& ctx) {
  auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
  auto blas = math::GetBlas<DeviceContext, T>(ctx);
  auto sigma_t = EigenTensor<T, 2>::From(*sigma);
  auto weight_t = EigenTensor<T, 2>::From(*weight);
  auto u_t = EigenTensor<T, 2>::From(*u);
  auto v_t = EigenTensor<T, 2>::From(*v);

  const int h = weight->dims()[0];
  const int w = weight->dims()[1];

  // LOG(ERROR) << "weight: " << weight_t;
  // LOG(ERROR) << "weight_trans: " << weight_trans_t;
  for (int i = 0; i < power_iters; i++) {
    // v_t.device(place) = weight_trans_t.contract(u_t, product_dims);
    blas.MatMul(*weight, true, *u, false, T(1), v, T(0));
    // LOG(ERROR) << "iter v: " << v_t;
    auto v_t_norm =
        v_t.square().sum().sqrt().eval().reshape(Array1(1)).broadcast(
            Array1(w));
    // LOG(ERROR) << "iter v_norm: " << v_t_norm;
    v_t.device(place) = v_t / (v_t_norm + v_t_norm.constant(eps));
    // LOG(ERROR) << "iter norm v: " << v_t;
    // u_t.device(place) = weight_t.contract(v_t, product_dims);
    blas.MatMul(*weight, false, *v, false, T(1), u, T(0));
    // LOG(ERROR) << "iter u: " << u_t;
    auto u_t_norm =
        u_t.square().sum().sqrt().eval().reshape(Array1(1)).broadcast(
            Array1(h));
    u_t.device(place) = u_t / (u_t_norm + u_t_norm.constant(eps));
    // LOG(ERROR) << "iter norm u: " << u_t;
  }
  // LOG(ERROR) << "h" << h << "w" << w;
  // LOG(ERROR) << "u: " << u_t;
  // LOG(ERROR) << "v: " << v_t;
  Tensor weight_v;
  weight_v.mutable_data<T>({h, 1}, ctx.GetPlace());
  blas.MatMul(*weight, false, *v, false, T(1), &weight_v, T(0));
  auto weight_v_t = EigenTensor<T, 2>::From(weight_v);
  // LOG(ERROR) << "weight_v: " << weight_v_t;
  sigma_t.device(place) = (u_t * weight_v_t)
                              .sum()
                              .eval()
                              .reshape(Array2(1, 1))
                              .broadcast(Array2(h, w));
  // LOG(ERROR) << "weight: " << weight_t;
  // LOG(ERROR) << "sigma: " << sigma_t;
  weight_t.device(place) = weight_t / sigma_t;
}

template <typename DeviceContext, typename T>
class SpectralNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto weight = ctx.Input<Tensor>("Weight");
    auto u = ctx.Input<Tensor>("U");
    auto v = ctx.Input<Tensor>("V");
    auto out = ctx.Output<Tensor>("Out");

    int dim = ctx.Attr<int>("dim");
    int power_iters = ctx.Attr<int>("power_iters");
    float eps = ctx.Attr<float>("eps");

    const int h = weight->dims()[0];
    const int w = weight->dims()[1];

    Tensor weight_mat;
    TensorCopySync(*weight, ctx.GetPlace(), &weight_mat);
    ResizeWeight(&weight_mat, dim);

    Tensor sigma;
    sigma.mutable_data<T>(weight->dims(), ctx.GetPlace());
    Tensor uu, vv;
    TensorCopySync(*u, ctx.GetPlace(), &uu);
    TensorCopySync(*v, ctx.GetPlace(), &vv);
    CalcMatrixSigmaAndNormWeight<DeviceContext, T>(
        &sigma, &(uu.Resize({h, 1})), &(vv.Resize({w, 1})), &weight_mat,
        power_iters, eps, ctx);
    TensorCopySync(weight_mat, ctx.GetPlace(), out);
  }
};

template <typename DeviceContext, typename T>
class SpectralNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle
