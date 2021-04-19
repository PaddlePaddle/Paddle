/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <boost/preprocessor/arithmetic/div.hpp>
#include <boost/preprocessor/arithmetic/mod.hpp>
#include <boost/preprocessor/comparison/greater.hpp>
#include <boost/preprocessor/comparison/greater_equal.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"

#define MAX_RANK_SUPPORTED 6
// 1. BOOST_PP_REPEAT macro represents a fast horizontal repetition construct.
//    Usage: BOOST_PP_REPEAT(count, macro, data).
//    This macro expands to the sequence:
//    macro(z, 0, data) macro(z, 1, data) ... macro(z, count - 1, data).
// 2. As for our case, count = MAX_RANK_SUPPORTED(which is 6).
//    So the range of n is 0-5(which is count-1).
//    We want to generate case 1-6 instead of case 0-5.
//    So we need to change n to n + 1.
#define EXPAND_AS_TEMPLATE(z, n, data) \
  case n + 1: {                        \
    ExpandAs<n + 1>(context);          \
    break;                             \
  }
#define REP_EXPAND_AS_TEMPLATE(n) BOOST_PP_REPEAT(n, EXPAND_AS_TEMPLATE, ~)
#define COND(n) BOOST_PP_GREATER_EQUAL(n, BOOST_PP_MOD(n, MAX_RANK_SUPPORTED))
#define EXPAND_AS_GRAD_CASE(n)                                           \
  case n + 1: {                                                          \
    ExpandAsBackward<n + 1>(context, reshape_dims_vec, reduce_dims_vec); \
    break;                                                               \
  }
#define EXPAND_AS_GRAD_TEMPLATE(z, n, data) \
  BOOST_PP_IF(COND(n), EXPAND_AS_GRAD_CASE(n), )
#define REP_EXPAND_AS_GRAD_TEMPLATE(n) \
  BOOST_PP_REPEAT(n, EXPAND_AS_GRAD_TEMPLATE, ~)

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class ExpandAsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto rank = context.Input<Tensor>("X")->dims().size();
    switch (rank) {
      REP_EXPAND_AS_TEMPLATE(MAX_RANK_SUPPORTED)
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Only support tensor with rank being between 1 and 6. But received "
            "tensor X's rank = %d.",
            rank));
    }
  }

 protected:
  template <int Rank>
  void ExpandAs(const framework::ExecutionContext& context) const {
    auto* in0 = context.Input<Tensor>("X");
    auto in_dims = in0->dims();
    auto* target_tensor = context.Input<Tensor>("target_tensor");
    auto* out0 = context.Output<Tensor>("Out");
    Eigen::DSizes<Eigen::DenseIndex, Rank> bcast_dims;
    int bcast_dims_remainder = 0;
    auto x_dims = in0->dims();
    auto y_dims = target_tensor->dims();
    for (int i = 0; i < y_dims.size(); ++i) {
      PADDLE_ENFORCE_NE(
          x_dims[i], 0UL,
          platform::errors::InvalidArgument(
              "X(input) should not have 0 dim. But received x_dims[%d] = 0.",
              i));
      bcast_dims[i] = y_dims[i] / x_dims[i];
      bcast_dims_remainder += y_dims[i] % x_dims[i];
    }
    PADDLE_ENFORCE_EQ(
        bcast_dims_remainder, 0UL,
        platform::errors::InvalidArgument(
            "X(input) could not be broadcast together with remapped "
            "shape(expand tensor's shape)"));
    framework::DDim out_dims(in_dims);
    for (size_t i = 0; i < bcast_dims.size(); ++i) {
      out_dims[i] *= bcast_dims[i];
    }

    out0->Resize(out_dims);
    auto x = EigenTensor<T, Rank>::From(*in0);
    out0->mutable_data<T>(context.GetPlace());
    auto y = EigenTensor<T, Rank>::From(*out0);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(place, y, x,
                                                                 bcast_dims);
  }
};

template <typename DeviceContext, typename T>
class ExpandAsGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<Tensor>("X");
    auto* target_tensor = context.Input<Tensor>("target_tensor");
    auto x_dims = in0->dims();
    auto y_dims = target_tensor->dims();
    std::vector<int> bcast_dims;
    for (int i = 0; i < y_dims.size(); ++i) {
      bcast_dims.push_back(y_dims[i] / x_dims[i]);
    }
    std::vector<int> reshape_dims_vec;
    std::vector<int> reduce_dims_vec;
    for (size_t i = 0; i < bcast_dims.size(); ++i) {
      reduce_dims_vec.push_back(reshape_dims_vec.size());
      reshape_dims_vec.push_back(bcast_dims[i]);
      reshape_dims_vec.push_back(x_dims[i]);
    }
    int dims = reduce_dims_vec.size();
    bool just_copy = true;
    for (size_t i = 0; i < bcast_dims.size(); i++) {
      if (bcast_dims[i] != 1) {
        just_copy = false;
        break;
      }
    }
    // no need reduce, just copy
    if (just_copy) {
      auto* in0 = context.Input<Tensor>(framework::GradVarName("Out"));
      auto* out0 = context.Output<Tensor>(framework::GradVarName("X"));
      out0->mutable_data<T>(context.GetPlace());
      framework::TensorCopy(*in0, context.GetPlace(), context.device_context(),
                            out0);
    } else {
      PADDLE_ENFORCE_GE(dims, 1,
                        platform::errors::InvalidArgument(
                            "The rank of the input 'Out@GRAD' for "
                            "expand_as_grad op must be greater than or "
                            "equal to 1, but the value received is %d.",
                            dims));
      PADDLE_ENFORCE_LE(dims, MAX_RANK_SUPPORTED,
                        platform::errors::InvalidArgument(
                            "The rank of the input 'Out@GRAD' for "
                            "expand_as_grad op must be less than or equal "
                            "to %d, but the value received is %d.",
                            MAX_RANK_SUPPORTED, dims));
      switch (dims) {
        REP_EXPAND_AS_GRAD_TEMPLATE(MAX_RANK_SUPPORTED)
        default:
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Only support tensor with rank being between 1 and 6. But "
              "received tensor's rank = %d.",
              dims));
      }
    }
  }

 protected:
  template <int Dims>
  void ExpandAsBackward(const framework::ExecutionContext& context,
                        const std::vector<int>& reshape_dims_vec,
                        const std::vector<int>& reduce_dims_vec) const {
    size_t reshape_size = reshape_dims_vec.size();
    size_t reduce_size = reduce_dims_vec.size();
    auto* in0 = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* out0 = context.Output<Tensor>(framework::GradVarName("X"));
    out0->mutable_data<T>(context.GetPlace());
    auto x_grad = EigenVector<T>::Flatten(*out0);
    Eigen::DSizes<Eigen::DenseIndex, Dims * 2> reshape_dims;
    for (size_t i = 0; i < reshape_size; ++i) {
      reshape_dims[i] = reshape_dims_vec[i];
    }
    Eigen::DSizes<Eigen::DenseIndex, Dims> reduce_dims;
    for (size_t i = 0; i < reduce_size; ++i) {
      reduce_dims[i] = reduce_dims_vec[i];
    }
    auto out_grad = EigenVector<T>::Flatten(*in0);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    EigenBroadcastGrad<std::decay_t<decltype(place)>, T, Dims>::Eval(
        place, x_grad, out_grad, reduce_dims, reshape_dims);
  }
};

}  // namespace operators
}  // namespace paddle
