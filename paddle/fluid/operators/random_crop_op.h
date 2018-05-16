// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/for_range.h"
#include "thrust/random.h"

namespace paddle {
namespace operators {

template <typename DeviceContext>
struct Random;

template <>
struct Random<platform::CPUDeviceContext> {
  using Engine = std::minstd_rand;

  template <typename T>
  using UniformIntDist = std::uniform_int_distribution<T>;
};

template <>
struct Random<platform::CUDADeviceContext> {
  using Engine = thrust::minstd_rand;

  template <typename T>
  using UniformIntDist = thrust::uniform_int_distribution<T>;
};

template <typename T>
HOSTDEVICE inline void RandomCropImpl(const T* x, size_t* x_dim, T* out,
                                      size_t* out_dim, int i, int rank,
                                      int64_t prod_x_remain,
                                      int64_t prod_out_remain, size_t* offset) {
  size_t x_length = x_dim[rank];
  size_t out_length = out_dim[rank];

  int64_t x_stride = prod_x_remain / x_length;
  int64_t out_stride = prod_out_remain / out_length;
  size_t offset_i = offset[i];
  if (x_stride == 1 && out_stride == 1) {
    // In the final stage, copy from offset.
    x += offset_i;
    for (size_t i = 0; i < out_length; ++i) {
      *out++ = *x++;
    }
  } else {
    x += offset_i * x_stride;
    for (size_t i = 0; i < out_length; ++i) {
      RandomCropImpl<T>(x, x_dim, out, out_dim, i + 1, rank, x_stride,
                        out_stride, offset);
      x += x_stride;
      out += out_stride;
    }
  }
}

template <typename DeviceContext, typename T>
struct RandomCropFunctor {
  const T* x_;
  T* out_;
  size_t x_dim_[9];
  size_t out_dim_[9];
  size_t prod_same_dim_;

  size_t prod_x_dim_;
  size_t prod_out_dim_;

  int num_same_dim_;
  int rank_;

  int64_t seed_;

  RandomCropFunctor(const T* x, T* out, int64_t seed)
      : x_(x),
        out_(out),
        prod_same_dim_(1),
        prod_x_dim_(1),
        prod_out_dim_(1),
        seed_(seed) {
    std::fill(x_dim_, x_dim_ + sizeof(x_dim_) / sizeof(size_t), 0);
    std::fill(out_dim_, out_dim_ + sizeof(out_dim_) / sizeof(size_t), 0);
  }

  HOSTDEVICE void operator()(size_t i) {
    typename Random<DeviceContext>::Engine engine(seed_);
    engine.discard(i * (rank_ - num_same_dim_));

    int64_t prod_x_unsame = (prod_x_dim_ / prod_same_dim_);
    int64_t prod_out_unsame = (prod_out_dim_ / prod_same_dim_);

    const T* x = x_ + i * prod_x_unsame;
    T* out = out_ + i * prod_out_unsame;

    size_t offset[9];
    for (int i = num_same_dim_; i < rank_; ++i) {
      typename Random<DeviceContext>::template UniformIntDist<size_t> dist(
          0, x_dim_[i] - out_dim_[i]);
      offset[i] = dist(engine);
    }
    RandomCropImpl<T>(x, x_dim_, out, out_dim_, num_same_dim_, rank_,
                      prod_x_unsame, prod_out_unsame, offset);
  }
};

template <typename DeviceContext, typename T>
class RandomCropKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& context) const {
    int64_t seed =
        *context.Input<framework::LoDTensor>("Seed")->data<int64_t>();
    auto& x = detail::Ref(context.Input<framework::LoDTensor>("X"));
    auto& out = detail::Ref(context.Output<framework::LoDTensor>("Out"));

    RandomCropFunctor<DeviceContext, T> functor{
        x.data<T>(), out.mutable_data<T>(context.GetPlace()), seed};

    auto& out_dim = out.dims();
    auto& x_dim = x.dims();

    auto rank = x_dim.size();
    while (rank-- > 0) {
      functor.x_dim_[rank] = x_dim[rank];
      functor.out_dim_[rank] = out_dim[rank];
      functor.prod_x_dim_ *= x_dim[rank];
      functor.prod_out_dim_ *= out_dim[rank];
      if (x_dim[rank] != out_dim[rank]) {
        PADDLE_ENFORCE_EQ(functor.prod_same_dim_, 1);
        functor.num_same_dim_ = rank;
      } else {
        functor.prod_same_dim_ *= out_dim[rank];
      }
    }
    functor.rank_ = x_dim.size();

    platform::ForRange<DeviceContext> for_range(
        context.template device_context<DeviceContext>(),
        functor.prod_same_dim_);

    for_range(functor);

    Random<platform::CPUDeviceContext>::Engine engine(seed);
    engine.discard(functor.prod_same_dim_ *
                   (functor.rank_ - functor.num_same_dim_));

    *context.Output<framework::LoDTensor>("SeedOut")->mutable_data<int64_t>(
        platform::CPUPlace()) = engine();
  }
};

}  // namespace operators
}  // namespace paddle
