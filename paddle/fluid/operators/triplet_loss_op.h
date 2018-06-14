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

template <typename DeviceContext, typename T>
class TripletLossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* labels = context.Input<Tensor>("Label");
    Tensor* loss = context.Output<Tensor>("Loss");
    Tensor* d_logits = context.Output<Tensor>("LogitsGrad");
    loss->mutable_data<T>(context.GetPlace());
    d_logits->mutable_data<T>(context.GetPlace());
    auto x_dims = logits->dims();
    int batch_size = x_dims[0];
    int feature_len = x_dims[1];

    // step 1: get distance matrix
    Tensor distances;
    distances.mutable_data<T>({batch_size, batch_size}, context.GetPlace());
    Tensor d_distances;
    d_distances.mutable_data<T>({batch_size, batch_size}, context.GetPlace());
    auto x_t = EigenMatrix<T>::From(*logits);
    auto d_x_t = EigenMatrix<T>::From(*d_logits);
    auto loss_t = EigenVector<T>::From(*loss);
    auto distances_t = EigenMatrix<T>::From(distances);
    auto d_distances_t = EigenMatrix<T>::From(d_distances);

    auto blas = math::GetBlas<DeviceContext, T>(context);
    auto x_mat = math::CreateMatrixDescriptor(x_dims, 0, false);
    auto x_mat_trans = math::CreateMatrixDescriptor(x_dims, 0, true);
    blas.MatMul(x_t, x_mat, x_t, x_mat_trans, T(1), distances, T(0));
    distances_t = x_t.square().sum(DIM1({1})).broadcast(DIM2({1, batch_size})) +
                  x_t.square()
                      .sum(DIM1({1}))
                      .shuffle(DIM2({1, 0}))
                      .broadcast(DIM2({batch_size, 1})) -
                  2 * D;

    // step 2: get loss in each line of distance matrix
    float eps = 0.5;
    auto relu = [eps](float x) { return x < 0 ? eps : x + eps; };
    auto relu_grad = [](float x) { return x < 0 ? 0 : 1; };
    auto offsets = GetOffsets<DeviceContext>(labels);
    for (size_t i = 0; i < offsets.size() - 1; ++i) {
      int begin = offsets[i];
      int end = offsets[i + 1];
      int pos_num = end - begin;
      int neg_num = batch_size - pos_num;
      for (size_t j = begin; j < end; ++j) {
        // get loss in current line
        auto p_dis = distances_t.slice(DIM2({j, begin}), DIM2({1, pos_num}));
        auto n_dis = distances_t.slice(DIM2({j, 0}), DIM2({1, batch_size}))
                         .reshape(DIM2({batch_size, 1}));
        auto n_p_sub =
            n_dis.broadcast(DIM2({pos_num, 1})) -
            p_dis.shuffle(DIM2({1, 0})).broadcast(DIM2({1, neg_num}));
        auto p_p_sub =
            p_dis.broadcast(DIM2({pos_num, 1})) -
            p_dis.shuffle(DIM2({1, 0})).broadcast(DIM2({1, pos_num}));
        loss_t(j) =
            n_p_sub.unaryExpr(relu).sum() - p_p_sub.unaryExpr(relu).sum();
        // get gradient of distance matric in current line
        d_distances_t.slice(DIM2({j, begin}), DIM2({1, pos_num})) =
            p_p_sub.unaryExpr(relu_grad).sum(DIM1({1})).shuffle(DIM2({1, 0})) -
            n_p_sub.unaryExpr(relu_grad).sum(DIM1({1})).shuffle(DIM2({1, 0})) -
            p_p_sub.unaryExpr(relu_grad).sum(DIM1({0}));
        d_distances_t.slice(DIM2({j, 0}), DIM2({1, batch_size})) =
            n_p_sub.unaryExpr(relu_grad).sum(DIM1({0}));
      }
      // get gradient of logits
      auto d_x_t_2_sum = d_distances_t.sum(DIM1({1})) +
                         d_distances_t.sum(DIM1({0})).shuffle(DIM2({1, 0}));
      auto d_x_t_2 = d_x_t_2_sum.broadcast(DIM2({1, feature_len}));
      d_x_t = 2 * x_t * d_x_t_2 + 2 * x_t;
    }
  }
};

template <typename DeviceContext, typename T>
class TripletLossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* loss = context.Input<Tensor>(framework::GradVarName("Loss"));
    const Tensor* d_logits = context.Input<Tensor>("LogitsGrad");
    Tensor* d_in = context.Output<Tensor>(framework::GradVarName("Logits"));
    d_in->mutable_data<T>(context.GetPlace());
    auto d_in_dims = d_in->dims();
    int batch_size = d_in_dims[0];
    int feature_len = d_in_dims[1] auto d_logits_t =
        EigenMatrix<T>::From(*d_logits);
    auto d_loss_t = EigenVector<T>::From(*d_loss);
    auto d_in_t = EigenMatrix<T>::From(*d_in);
    // d_in = d_logis * d_loss
    for (size_t i = 0; i < batch_size; ++i) {
      d_in_t.slice(DIM2({i, 0}), DIM2({1, feature_len})) =
          d_logits_t.slice(DIM2({i, 0}), DIM2({1, feature_len})) * d_loss_t(i);
    }
  }
};

}  // namespace operators
}  // namespace paddle
