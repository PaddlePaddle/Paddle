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

static inline void CalcMatrixShape(const Tensor& weight, const int dim, int* h,
                                   int* w) {
  auto weight_dims = weight.dims();
  *h = 1;
  *w = 1;
  for (int i = 0; i < weight_dims.size(); i++) {
    if (i <= dim) {
      *h *= weight_dims[i];
    } else {
      *w *= weight_dims[i];
    }
  }
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

  for (int i = 0; i < power_iters; i++) {
    blas.MatMul(*weight, true, *u, false, T(1), v, T(0));
    auto v_t_norm =
        v_t.square().sum().sqrt().eval().reshape(Array1(1)).broadcast(
            Array1(w));
    v_t.device(place) = v_t / (v_t_norm + v_t_norm.constant(eps));
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

    Tensor weight_mat;
    int h, w;
    CalcMatrixShape(*weight, dim, &h, &w);
    TensorCopySync(*weight, ctx.GetPlace(), &weight_mat);
    weight_mat = weight_mat.Resize({h, w});

    Tensor sigma;
    sigma.mutable_data<T>(weight_mat.dims(), ctx.GetPlace());
    Tensor uu, vv;
    TensorCopySync(*u, ctx.GetPlace(), &uu);
    TensorCopySync(*v, ctx.GetPlace(), &vv);
    CalcMatrixSigmaAndNormWeight<DeviceContext, T>(
        &sigma, &(uu.Resize({h, 1})), &(vv.Resize({w, 1})), &weight_mat,
        power_iters, eps, ctx);
    TensorCopySync(weight_mat.Resize(out->dims()), ctx.GetPlace(), out);
  }
};

template <typename DeviceContext, typename T>
class SpectralNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    auto weight = ctx.Input<Tensor>("Weight");
    auto u = ctx.Input<Tensor>("U");
    auto v = ctx.Input<Tensor>("V");
    auto out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto weight_grad = ctx.Output<Tensor>(framework::GradVarName("Weight"));

    int dim = ctx.Attr<int>("dim");
    int power_iters = ctx.Attr<int>("power_iters");
    float eps = ctx.Attr<float>("eps");

    Tensor weight_mat, out_grad_mat;
    int h, w;
    CalcMatrixShape(*weight, dim, &h, &w);
    TensorCopySync(*weight, ctx.GetPlace(), &weight_mat);
    TensorCopySync(*out_grad, ctx.GetPlace(), &out_grad_mat);
    weight_mat = weight_mat.Resize({h, w});
    out_grad_mat = out_grad_mat.Resize({h, w});

    Tensor sigma;
    sigma.mutable_data<T>(weight_mat.dims(), ctx.GetPlace());
    Tensor uu, vv;
    TensorCopySync(*u, ctx.GetPlace(), &uu);
    TensorCopySync(*v, ctx.GetPlace(), &vv);
    CalcMatrixSigmaAndNormWeight<DeviceContext, T>(
        &sigma, &(uu.Resize({h, 1})), &(vv.Resize({w, 1})), &weight_mat,
        power_iters, eps, ctx);

    Tensor uv;
    uv.mutable_data<T>({h, w}, ctx.GetPlace());
    blas.MatMul(uu.Resize({h, 1}), false, vv.Resize({w, 1}), false, T(1), &uv,
                T(0));

    Tensor weight_grad_mat, ones;
    weight_grad_mat.mutable_data<T>({h, w}, ctx.GetPlace());
    ones.mutable_data<T>({h, w}, ctx.GetPlace());
    auto weight_grad_mat_t = EigenTensor<T, 2>::From(weight_grad_mat);
    auto weight_mat_t = EigenTensor<T, 2>::From(weight_mat);
    auto out_grad_mat_t = EigenTensor<T, 2>::From(out_grad_mat);
    auto sigma_t = EigenTensor<T, 2>::From(sigma);
    auto uv_t = EigenTensor<T, 2>::From(uv);
    auto ones_t = EigenTensor<T, 2>::From(ones).setConstant((T)1);
    weight_mat_t.device(place) =
        weight_mat_t.sum().eval().reshape(Array2(1, 1)).broadcast(Array2(h, w));
    weight_grad_mat_t.device(place) =
        out_grad_mat_t * (ones_t - uv_t * weight_mat_t) / sigma_t;
    TensorCopySync(weight_grad_mat.Resize(weight_grad->dims()), ctx.GetPlace(),
                   weight_grad);
  }
};

}  // namespace operators
}  // namespace paddle
