// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include <boost/preprocessor/arithmetic/mod.hpp>
#include <boost/preprocessor/comparison/greater.hpp>
#include <boost/preprocessor/comparison/greater_equal.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/platform/errors.h"

#define MAX_RANK_SUPPORTED 6
// 1. BOOST_PP_REPEAT macro represents a fast horizontal repetition construct.
//    Usage: BOOST_PP_REPEAT(count, macro, data).
//    This macro expands to the sequence:
//    macro(z, 0, data) macro(z, 1, data) ... macro(z, count - 1, data).
// 2. As for our case, count = MAX_RANK_SUPPORTED(which is 6).
//    So the range of n is 0-5(which is count-1).
//    We want to generate case 1-6 instead of case 0-5.
//    So we need to change n to n + 1.
#define MESHGRID_TEMPLATE(z, n, data) \
  case n + 1: {                       \
    MeshgridForward<n + 1>(context);  \
    break;                            \
  }
#define REP_MESHGRID_TEMPLATE(n) BOOST_PP_REPEAT(n, MESHGRID_TEMPLATE, ~)
#define COND(n) BOOST_PP_GREATER_EQUAL(n, BOOST_PP_MOD(n, MAX_RANK_SUPPORTED))

#define MESHGRID_GRAD_CASE(n)         \
  case n + 1: {                       \
    MeshgridBackward<n + 1>(context); \
    break;                            \
  }
#define MESHGRID_GRAD_TEMPLATE(z, n, data) \
  BOOST_PP_IF(COND(n), MESHGRID_GRAD_CASE(n), )
#define REP_MESHGRID_GRAD_TEMPLATE(n) \
  BOOST_PP_REPEAT(n, MESHGRID_GRAD_TEMPLATE, ~)

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MeshgridKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto ins = context.MultiInput<framework::Tensor>("X");
    auto rank = ins.size();
    switch (rank) {
      REP_MESHGRID_TEMPLATE(MAX_RANK_SUPPORTED)
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Excepted Tensor numbers between 1 and 6, but only received d% .",
            rank));
    }
  }

 protected:
  template <int Rank>
  void MeshgridForward(const framework::ExecutionContext& context) const {
    auto ins = context.MultiInput<framework::Tensor>("X");
    auto outs = context.MultiOutput<framework::Tensor>("Out");
    PADDLE_ENFORCE_EQ(
        ins.size() > 1, true,
        platform::errors::InvalidArgument(
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
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Expected scalar or 1D tensor in the tensor list but got tensor "
              "%d: ",
              i));
      }
    }

    for (int64_t i = 0; i < size; i++) {
      std::vector<int64_t> view_shape(size, 1);
      view_shape[i] = shape[i];

      framework::Tensor reshape_ins_tensor;
      TensorCopy(*ins[i], context.GetPlace(), context.device_context(),
                 &reshape_ins_tensor);
      framework::DDim out_dims_reshape = framework::make_ddim(view_shape);
      reshape_ins_tensor.Resize(out_dims_reshape);
      framework::DDim out_dims = framework::make_ddim(shape);

      Eigen::DSizes<Eigen::DenseIndex, Rank> bcast_dims;
      for (int64_t j = 0; j < size; j++) {
        bcast_dims[j] = shape[j];
      }
      bcast_dims[i] = 1;

      outs[i]->Resize(out_dims);
      auto x = framework::EigenTensor<T, Rank>::From(
          static_cast<const framework::Tensor>(reshape_ins_tensor));
      outs[i]->mutable_data<T>(context.GetPlace());
      auto y = framework::EigenTensor<T, Rank>::From(*outs[i]);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(place, y, x,
                                                                   bcast_dims);
    }
  }
};

template <typename DeviceContext, typename T>
class MeshgridGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto out_grad =
        context.MultiInput<framework::Tensor>(framework::GradVarName("Out"));
    int n = out_grad.size();
    switch (n) {
      REP_MESHGRID_GRAD_TEMPLATE(MAX_RANK_SUPPORTED)
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Excepted Tensor numbers between 1 and 6, but only received d% .",
            n));
    }
  }

 protected:
  template <int Rank>
  void MeshgridBackward(const framework::ExecutionContext& context) const {
    auto out_grad =
        context.MultiInput<framework::Tensor>(framework::GradVarName("Out"));
    auto ins = context.MultiInput<framework::Tensor>("X");
    auto outs =
        context.MultiOutput<framework::Tensor>(framework::GradVarName("X"));

    int n = out_grad.size();
    auto out_dims = out_grad[0]->dims();

    for (int i = 0; i < n; i++) {
      outs[i]->mutable_data<T>(context.GetPlace());
      auto out_grad_tmp = framework::EigenVector<T>::Flatten(*out_grad[i]);
      auto in_grad = framework::EigenVector<T>::Flatten(*outs[i]);

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

      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      EigenBroadcastGrad<std::decay_t<decltype(place)>, T, Rank>::Eval(
          place, in_grad, out_grad_tmp, reduce_dims, reshape_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle
