/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
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

#define MAX_RANK_SUPPORTED 6

#define TILE_TEMPLATE(z, n, data) \
  case n + 1: {                   \
    Tile<n + 1>(context);         \
    break;                        \
  }
#define REP_TILE_TEMPLATE(n) BOOST_PP_REPEAT(n, TILE_TEMPLATE, ~)
#define COND(n) BOOST_PP_GREATER_EQUAL(n, BOOST_PP_MOD(n, MAX_RANK_SUPPORTED))
#define TILE_GRAD_CASE(n)                                        \
  case n: {                                                      \
    TileBackward<n>(context, reshape_dims_vec, reduce_dims_vec); \
    break;                                                       \
  }
#define TILE_GRAD_TEMPLATE(z, n, data) BOOST_PP_IF(COND(n), TILE_GRAD_CASE(n), )
#define REP_TILE_GRAD_TEMPLATE(n) BOOST_PP_REPEAT(n, TILE_GRAD_TEMPLATE, ~)

namespace paddle {
namespace operators {
inline std::vector<int> get_repeat_times(
    const framework::ExecutionContext& ctx) {
  if (ctx.HasInput("RepeatTimes")) {
    auto* repeat_tensor = ctx.Input<framework::LoDTensor>("RepeatTimes");
    auto* repeat_data = repeat_tensor->data<int>();
    framework::Tensor cpu_repeat_tensor;
    if (platform::is_gpu_place(repeat_tensor->place())) {
      TensorCopySync(*repeat_tensor, platform::CPUPlace(), &cpu_repeat_tensor);
      repeat_data = cpu_repeat_tensor.data<int>();
    }
    auto vec_repeat_times =
        std::vector<int>(repeat_data, repeat_data + repeat_tensor->numel());
    return vec_repeat_times;
  }

  auto list_repeat_times_tensor =
      ctx.MultiInput<framework::Tensor>("repeat_times_tensor");
  if (list_repeat_times_tensor.size() > 0) {
    // get tensor from
    std::vector<int> vec_repeat_times;
    for (size_t i = 0; i < list_repeat_times_tensor.size(); ++i) {
      auto tensor = list_repeat_times_tensor[i];
      if (platform::is_gpu_place(tensor->place())) {
        framework::Tensor temp;
        TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_repeat_times.push_back(*temp.data<int32_t>());
      } else {
        vec_repeat_times.push_back(*tensor->data<int32_t>());
      }
    }
    return vec_repeat_times;
  } else {
    return ctx.Attr<std::vector<int>>("repeat_times");
  }
}

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using framework::To32BitIndex;

template <typename DeviceContext, typename T>
class TileKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto rank = context.Input<Tensor>("X")->dims().size();
    PADDLE_ENFORCE_GE(
        rank, 1, platform::errors::InvalidArgument(
                     "The rank of the input 'x' for tile op must be a positive "
                     "integer, but the value received is %d.",
                     rank));
    PADDLE_ENFORCE_LE(
        rank, MAX_RANK_SUPPORTED,
        platform::errors::InvalidArgument(
            "The rank of the input 'x' for tile op "
            "must be less than or equal to %d, but the value received is %d.",
            MAX_RANK_SUPPORTED, rank));
    auto repeat_times = get_repeat_times(context);
    int repeat_times_size = repeat_times.size();
    PADDLE_ENFORCE_GE(
        repeat_times_size, 1,
        platform::errors::InvalidArgument(
            "The number of elements of the input 'repeat_times' for tile "
            "op must be positive, but the value received is %d.",
            repeat_times_size));
    PADDLE_ENFORCE_LE(
        repeat_times_size, MAX_RANK_SUPPORTED,
        platform::errors::InvalidArgument(
            "The number of elements of the input 'repeat_times' for tile op "
            "must be less than or equal to %d, but the value received is %d.",
            MAX_RANK_SUPPORTED, repeat_times_size));
    rank = std::max(rank, repeat_times_size);
    switch (rank) { REP_TILE_TEMPLATE(MAX_RANK_SUPPORTED) }
  }

 protected:
  template <int Rank>
  void Tile(const framework::ExecutionContext& context) const {
    auto* in0 = context.Input<Tensor>("X");

    auto in_dims = in0->dims();
    auto repeat_times = get_repeat_times(context);
    for (size_t i = 0; i < repeat_times.size(); ++i) {
      PADDLE_ENFORCE_GT(
          repeat_times[i], 0,
          platform::errors::InvalidArgument(
              "All elements of the input 'repeat_times' for tile op must "
              "be positive integers, but the value received is %d.",
              repeat_times[i]));
    }
    auto vec_in_dims = framework::vectorize<int>(in_dims);
    if (repeat_times.size() < vec_in_dims.size()) {
      int diff = vec_in_dims.size() - repeat_times.size();
      repeat_times.insert(repeat_times.begin(), diff, 1);
    } else {
      int diff = repeat_times.size() - vec_in_dims.size();
      vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
    }
    PADDLE_ENFORCE_EQ(
        repeat_times.size(), vec_in_dims.size(),
        platform::errors::InvalidArgument(
            "The rank (%d) of the input 'x' and the rank (%d) of the input "
            "'repeat_times' for tile op must match after promotion.",
            vec_in_dims.size(), repeat_times.size()));
    auto* out0 = context.Output<Tensor>("Out");
    Eigen::DSizes<int, Rank> bcast_dims;
    for (size_t i = 0; i < repeat_times.size(); ++i) {
      bcast_dims[i] = repeat_times[i];
    }

    framework::DDim new_in_dims = framework::make_ddim(vec_in_dims);
    framework::DDim out_dims(new_in_dims);
    for (size_t i = 0; i < repeat_times.size(); ++i) {
      out_dims[i] *= repeat_times[i];
    }

    out0->Resize(out_dims);
    auto x = EigenTensor<T, Rank>::From(*in0, new_in_dims);
    out0->mutable_data<T>(context.GetPlace());
    auto y = EigenTensor<T, Rank>::From(*out0, out_dims);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    // use 32-bit index to speed up
    bool use_32bit_index = y.size() < Eigen::NumTraits<int>::highest();
    if (use_32bit_index) {
      To32BitIndex(y).device(place) = To32BitIndex(x).broadcast(bcast_dims);
    } else {
      y.device(place) = x.broadcast(bcast_dims);
    }
  }
};

template <typename DeviceContext, typename T>
class TileGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto repeat_times = get_repeat_times(context);
    auto x_dims = x->dims();
    auto vec_in_dims = framework::vectorize<int>(x_dims);
    if (repeat_times.size() < vec_in_dims.size()) {
      int diff = vec_in_dims.size() - repeat_times.size();
      repeat_times.insert(repeat_times.begin(), diff, 1);
    } else {
      int diff = repeat_times.size() - vec_in_dims.size();
      vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
    }
    // 1. reshape_dims_vec is the broadcast parameter.
    // 2. reduce_dims_vec is the dimension parameter to compute gradients. For
    //    each dimension expanded, the gradients should be summed to original
    //    size.
    std::vector<int> reshape_dims_vec;
    std::vector<int> reduce_dims_vec;
    for (size_t i = 0; i < repeat_times.size(); ++i) {
      reduce_dims_vec.push_back(reshape_dims_vec.size());
      reshape_dims_vec.push_back(repeat_times[i]);
      reshape_dims_vec.push_back(vec_in_dims[i]);
    }

    int dims = reduce_dims_vec.size();

    bool just_copy = true;
    for (size_t i = 0; i < repeat_times.size(); i++) {
      if (repeat_times[i] != 1) {
        just_copy = false;
        break;
      }
    }
    // no need reduce, just copy
    if (just_copy) {
      auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
      auto* dx = context.Output<Tensor>(framework::GradVarName("X"));
      dx->mutable_data<T>(context.GetPlace());
      framework::TensorCopy(*dout, context.GetPlace(), context.device_context(),
                            dx);
      // TensorCopy may change the dims of dx
      dx->Resize(x_dims);
    } else {
      PADDLE_ENFORCE_GE(dims, 1,
                        platform::errors::InvalidArgument(
                            "Th rank of the input 'Out@GRAD' for tile_grad op "
                            " must be greater than or equal to 1, but "
                            "the value received is %d.",
                            dims));
      PADDLE_ENFORCE_LE(dims, MAX_RANK_SUPPORTED,
                        platform::errors::InvalidArgument(
                            "The rank of the input 'Out@GRAD' for tile_grad op "
                            "must be less than or equal "
                            "to %d, but the value received is %d.",
                            MAX_RANK_SUPPORTED, dims));
      switch (dims) { REP_TILE_GRAD_TEMPLATE(MAX_RANK_SUPPORTED) }
    }
  }

 protected:
  template <int Dims>
  void TileBackward(const framework::ExecutionContext& context,
                    const std::vector<int>& reshape_dims_vec,
                    const std::vector<int>& reduce_dims_vec) const {
    size_t reshape_size = reshape_dims_vec.size();
    size_t reduce_size = reduce_dims_vec.size();
    auto* in0 = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* out0 = context.Output<Tensor>(framework::GradVarName("X"));
    out0->mutable_data<T>(context.GetPlace());
    auto x_grad = EigenVector<T>::Flatten(*out0);
    Eigen::DSizes<int, Dims * 2> reshape_dims;
    for (size_t i = 0; i < reshape_size; ++i) {
      reshape_dims[i] = reshape_dims_vec[i];
    }
    Eigen::DSizes<int, Dims> reduce_dims;
    for (size_t i = 0; i < reduce_size; ++i) {
      reduce_dims[i] = reduce_dims_vec[i];
    }

    auto out_grad = EigenVector<T>::Flatten(*in0);
    x_grad.device(
        *context.template device_context<DeviceContext>().eigen_device()) =
        out_grad.reshape(reshape_dims)
            .sum(reduce_dims)
            .reshape(x_grad.dimensions());
  }
};

}  // namespace operators
}  // namespace paddle
