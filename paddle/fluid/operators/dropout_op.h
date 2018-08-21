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

#include <random>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/fluid/platform/random.h"

namespace paddle {
namespace operators {
using framework::Tensor;
using framework::EigenMatrix;

template <typename T, typename ProbType>
class FillMaskAndY {
 public:
  FillMaskAndY(T *mask, const T *x, T *y, ProbType prob)
      : mask_(mask), x_(x), y_(y), prob_(prob) {}

  HOSTDEVICE inline void operator()(size_t i, ProbType prob) {
    if (prob < prob_) {
      mask_[i] = static_cast<T>(0);
      y_[i] = static_cast<T>(0);
    } else {
      mask_[i] = static_cast<T>(1);
      y_[i] = x_[i];
    }
  }

 private:
  T *mask_;
  const T *x_;
  T *y_;
  ProbType prob_;
};

template <typename T>
class FillZeroYAndMask {
 public:
  FillZeroYAndMask(T *mask, T *y) : mask_(mask), y_(y) {}
  HOSTDEVICE inline void operator()(size_t i) {
    mask_[i] = static_cast<T>(0);
    y_[i] = static_cast<T>(0);
  }

 private:
  T *mask_;
  T *y_;
};

template <typename DeviceContext, typename T>
class DropoutKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<Tensor>("X");
    auto *y = context.Output<Tensor>("Out");
    const auto *x_data = x->data<T>();
    auto *y_data = y->mutable_data<T>(context.GetPlace());
    float dropout_prob = context.Attr<float>("dropout_prob");

    if (!context.Attr<bool>("is_test")) {
      auto *mask = context.Output<Tensor>("Mask");
      auto *mask_data = mask->mutable_data<T>(context.GetPlace());

      // NOTE: fixed seed should only be used in unittest or for debug.
      // Guarantee to use random seed in training.
      int seed = context.Attr<bool>("fix_seed") ? context.Attr<int>("seed")
                                                : std::random_device()();

      uint64_t uint32_prob =
          static_cast<uint64_t>(static_cast<uint64_t>(UINT32_MAX) *
                                static_cast<double>(dropout_prob));

      uint64_t uint16_prob =
          static_cast<uint64_t>(static_cast<uint64_t>(UINT16_MAX) *
                                static_cast<double>(dropout_prob));
      uint64_t uint8_prob = static_cast<uint64_t>(
          static_cast<uint64_t>(UINT8_MAX) * static_cast<double>(dropout_prob));

      if (uint32_prob >=
          (1UL << 32)) {  // speical case. Dropout rate = 1. always fill zero.
        // Fill Zero
        platform::ForRange<DeviceContext> for_range(
            context.template device_context<DeviceContext>(), x->numel());
        FillZeroYAndMask<T> fill_functor(mask_data, y_data);
        for_range(fill_functor);
        return;
      }

      constexpr double epsilon = 1e-3;
      auto uint8_err = std::abs((static_cast<double>(uint8_prob) / UINT8_MAX) -
                                dropout_prob);
      if (uint8_err < epsilon) {
        VLOG(10) << "Fast dropout use uint8_t";
        DropoutImpl<uint8_t>(context, x, x_data, y_data, mask_data, seed,
                             uint8_prob);
        return;
      }

      auto uint16_err = std::abs(
          (static_cast<double>(uint16_prob) / UINT16_MAX) - dropout_prob);
      if (uint16_err < epsilon) {
        VLOG(10) << "Fast dropout use uint16_t";
        DropoutImpl<uint16_t>(context, x, x_data, y_data, mask_data, seed,
                              uint16_prob);
        return;
      }
      // Fall back to uint32_t if the err is always larger than epsilon.
      DropoutImpl<uint32_t>(context, x, x_data, y_data, mask_data, seed,
                            uint32_prob);

    } else {
      auto &place =
          *context.template device_context<DeviceContext>().eigen_device();
      auto X = EigenMatrix<T>::Reshape(*x, 1);
      auto Y = EigenMatrix<T>::Reshape(*y, 1);
      Y.device(place) = X * static_cast<T>(1.0f - dropout_prob);
    }
  }

  template <typename ProbType>
  void DropoutImpl(const framework::ExecutionContext &context, const Tensor *x,
                   const T *x_data, T *y_data, T *mask_data, int seed,
                   uint64_t uint32_prob) const {
    FillMaskAndY<T, ProbType> fill_functor(mask_data, x_data, y_data,
                                           static_cast<ProbType>(uint32_prob));
    platform::random::RandomSequence<DeviceContext> rand_seq;
    platform::random::IdentityDistribution<ProbType> dist;
    rand_seq(context.template device_context<DeviceContext>(), seed, x->numel(),
             dist, fill_functor);
  }
};

template <typename DeviceContext, typename T>
class DropoutGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    PADDLE_ENFORCE(!context.Attr<bool>("is_test"),
                   "GradOp is only callable when is_test is false");

    auto *grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto *grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto *mask = context.Input<Tensor>("Mask");
    grad_x->mutable_data<T>(context.GetPlace());

    auto M = EigenMatrix<T>::Reshape(*mask, 1);
    auto dX = EigenMatrix<T>::Reshape(*grad_x, 1);
    auto dY = EigenMatrix<T>::Reshape(*grad_y, 1);

    auto &place =
        *context.template device_context<DeviceContext>().eigen_device();
    dX.device(place) = dY * M;
  }
};

}  // namespace operators
}  // namespace paddle
