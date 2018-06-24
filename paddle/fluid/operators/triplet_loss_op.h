/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

using DIM1 = Eigen::array<int, 1>;
using DIM2 = Eigen::array<int, 2>;

template <typename DeviceContext>
std::vector<int> GetOffsets(const Tensor* t);

template <typename T>
class ReluFunctor {
 public:
  explicit ReluFunctor(T eps) : eps_(eps) {}
  HOSTDEVICE T operator()(const T& x) const {
    if ((x + eps_) > 0)
      return x + eps_;
    else
      return 0;
  }

 private:
  T eps_;
};

template <typename T>
class ReluGradFunctor {
 public:
  HOSTDEVICE T operator()(const T& x) const { return x < 0 ? T(0) : T(1); }
};

template <typename DeviceContext, typename T>
class TripletLossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* labels = context.Input<Tensor>("Label");
    T eps = static_cast<T>(context.Attr<float>("epsilon"));
    Tensor* loss = context.Output<Tensor>("Loss");
    Tensor* d_logits = context.Output<Tensor>("LogitsGrad");
    loss->mutable_data<T>(context.GetPlace());
    d_logits->mutable_data<T>(context.GetPlace());
    logits->data<T>();
    auto x_dims = logits->dims();
    int batch_size = x_dims[0];
    int feature_len = x_dims[1];

    // step 1: get distance matrix
    Tensor distances;
    distances.mutable_data<T>({batch_size, batch_size}, context.GetPlace());
    Tensor d_distances;
    d_distances.mutable_data<T>({batch_size, batch_size}, context.GetPlace());
    Tensor tmp;
    tmp.mutable_data<T>({batch_size, batch_size}, context.GetPlace());
    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    set_zero(dev_ctx, &d_distances, static_cast<T>(0));

    auto logits_t = EigenMatrix<T>::From(*logits);
    auto d_logits_t = EigenMatrix<T>::From(*d_logits);
    auto loss_t = EigenMatrix<T>::From(*loss);
    auto distances_t = EigenMatrix<T>::From(distances);
    auto d_distances_t = EigenMatrix<T>::From(d_distances);
    auto tmp_t = EigenMatrix<T>::From(tmp);

    auto blas = math::GetBlas<DeviceContext, T>(context);
    auto x_mat = math::CreateMatrixDescriptor(x_dims, 0, false);
    auto x_mat_trans = math::CreateMatrixDescriptor(x_dims, 0, true);
    blas.MatMul(*logits, x_mat, *logits, x_mat_trans, T(1), &distances, T(0));
    auto a = logits_t.square()
                 .sum(DIM1({1}))
                 .reshape(DIM2({batch_size, 1}))
                 .broadcast(DIM2({1, batch_size}));
    auto b = a.shuffle(DIM2({1, 0}));
    distances_t.device(place) = a + b - distances_t * T(2.0);

    // step 2: get loss in each line of distance matrix
    ReluGradFunctor<T> relu_grad;
    ReluFunctor<T> relu(eps);
    auto offsets = GetOffsets<DeviceContext>(labels);
    for (size_t i = 0; i < offsets.size() - 1; ++i) {
      int begin = offsets[i];
      int end = offsets[i + 1];
      int pos_num = end - begin;
      for (int j = begin; j < end; ++j) {
        // get loss in current line
        auto p_dis = distances_t.slice(DIM2({j, begin}), DIM2({1, pos_num}))
                         .reshape(DIM2{1, pos_num});
        auto n_dis = distances_t.chip(j, 0).reshape(DIM2({1, batch_size}));
        auto n_p_sub =
            n_dis.broadcast(DIM2({pos_num, 1})) -
            p_dis.reshape(DIM2{pos_num, 1}).broadcast(DIM2({1, batch_size}));
        auto p_p_sub =
            p_dis.broadcast(DIM2({pos_num, 1})) -
            p_dis.shuffle(DIM2({1, 0})).broadcast(DIM2({1, pos_num}));

        loss_t.chip(j, 0).device(place) =
            n_p_sub.unaryExpr(relu).sum() - p_p_sub.unaryExpr(relu).sum();
        // get gradient of distance matric in current line
        d_distances_t.chip(j, 0).device(place) =
            n_p_sub.unaryExpr(relu_grad).sum(DIM1({0})).reshape(
                DIM2({1, batch_size}));

        d_distances_t.slice(DIM2({j, begin}), DIM2({1, pos_num}))
            .device(place) =
            p_p_sub.unaryExpr(relu_grad).sum(DIM1({1})).reshape(
                DIM2({1, pos_num})) -
            n_p_sub.unaryExpr(relu_grad).sum(DIM1({1})).reshape(
                DIM2({1, pos_num}));
      }
    }

    // get gradient of logits
    tmp_t = d_distances_t + d_distances_t.shuffle(DIM2({1, 0}));
    auto dis_mat =
        math::CreateMatrixDescriptor({batch_size, batch_size}, 0, false);
    blas.MatMul(tmp, dis_mat, *logits, x_mat, T(-2), d_logits, T(0));

    auto sub_grad = tmp_t.sum(DIM1{1})
                        .reshape(DIM2({batch_size, 1}))
                        .broadcast(DIM2({1, feature_len})) *
                    logits_t * T(2.0);
    auto result = d_logits_t + sub_grad;
    d_logits_t.device(place) = result;
  }
};

template <typename DeviceContext, typename T>
class TripletLossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    const Tensor* d_loss =
        context.Input<Tensor>(framework::GradVarName("Loss"));
    d_loss->data<T>();
    const Tensor* d_logits = context.Input<Tensor>("LogitsGrad");
    Tensor* d_in = context.Output<Tensor>(framework::GradVarName("Logits"));
    d_in->mutable_data<T>(context.GetPlace());
    auto d_in_dims = d_in->dims();
    int batch_size = d_in_dims[0];
    int feature_len = d_in_dims[1];
    auto d_logits_t = EigenMatrix<T>::From(*d_logits);
    auto d_loss_t = EigenMatrix<T>::From(*d_loss, {batch_size, 1});
    auto d_in_t = EigenMatrix<T>::From(*d_in);
    d_in_t.device(place) =
        d_logits_t * d_loss_t.broadcast(DIM2({1, feature_len}));
  }
};

}  // namespace operators
}  // namespace paddle
