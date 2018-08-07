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

namespace paddle {
namespace operators {

// It seems that if we include <boost/functional/hash.hpp>, there may be some
// compilation conflict with STL
// So we implement boost::hash_combine() here
inline HOSTDEVICE size_t HashCombine(size_t seed, size_t idx) {
  // use boost::hash_combine() to make seed more random
  seed ^= idx + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}

template <typename DeviceContext, typename T, typename DropoutFunctor>
class DropoutKernel : public framework::OpKernel<T> {
  using Tensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Output<Tensor>("Out");
    const auto* x_data = x->data<T>();
    auto* y_data = y->mutable_data<T>(context.GetPlace());
    float dropout_prob = context.Attr<float>("dropout_prob");

    if (!context.Attr<bool>("is_test")) {
      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<uint8_t>(context.GetPlace());

      bool fix_seed = context.Attr<bool>("fix_seed");
      size_t seed;
      if (fix_seed) {
        auto* seed_tensor = context.Input<Tensor>("SeedIn");
        if (seed_tensor->IsInitialized()) {
          if (platform::is_gpu_place(seed_tensor->place())) {
            LOG(WARNING)
                << "It is slow to place seed in GPU memory. Please verify "
                   "your program";
            Tensor cpu_seed;
            framework::TensorCopySync(*seed_tensor, platform::CPUPlace(),
                                      &cpu_seed);
            seed = static_cast<size_t>(cpu_seed.data<int64_t>()[0]);
          } else {
            seed = static_cast<size_t>(seed_tensor->data<int64_t>()[0]);
          }
        } else {
          seed = static_cast<size_t>(context.Attr<int>("startup_seed"));
        }
      } else {
        seed = static_cast<size_t>(std::random_device()());
      }

      size_t size = x->numel();
      DropoutFunctor functor;
      functor(context.template device_context<DeviceContext>(), x_data, y_data,
              mask_data, size, dropout_prob, seed);

      if (fix_seed) {
        seed = HashCombine(
            seed, static_cast<uint32_t>(static_cast<uint32_t>(-1) *
                                        static_cast<double>(dropout_prob)));
        seed = HashCombine(seed, size);
        auto* seed_out =
            context.Output<Tensor>("SeedOut")->mutable_data<int64_t>(
                platform::CPUPlace());
        *seed_out = static_cast<int64_t>(seed);
      }
    } else {
      auto dim = framework::make_ddim({x->numel()});
      auto X = framework::EigenTensor<T, 1>::From(*x, dim);
      auto Y = framework::EigenTensor<T, 1>::From(*y, dim);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      Y.device(place) = X * static_cast<T>(1.0f - dropout_prob);
    }
  }
};

template <typename DeviceContext, typename T>
class DropoutGradKernel : public framework::OpKernel<T> {
  using Tensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(!context.Attr<bool>("is_test"),
                   "GradOp is only callable when is_test is false");

    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* mask = context.Input<Tensor>("Mask");
    grad_x->mutable_data<T>(context.GetPlace());

    auto dim = framework::make_ddim({grad_y->numel()});
    auto M = framework::EigenTensor<uint8_t, 1>::From(*mask, dim);
    auto dX = framework::EigenTensor<T, 1>::From(*grad_x, dim);
    auto dY = framework::EigenTensor<T, 1>::From(*grad_y, dim);

    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    dX.device(place) = dY * M;
  }
};

struct CPUDropoutFunctor {
  template <typename T, typename MaskType>
  void operator()(const platform::CPUDeviceContext& ctx, const T* x_data,
                  T* y_data, MaskType* mask_data, size_t size,
                  float dropout_prob, size_t seed) {
    std::minstd_rand engine;
    engine.seed(seed);
    std::uniform_real_distribution<float> dist(0, 1);
    for (size_t i = 0; i < size; ++i) {
      if (dist(engine) < dropout_prob) {
        mask_data[i] = static_cast<MaskType>(0);
        y_data[i] = static_cast<T>(0);
      } else {
        mask_data[i] = static_cast<MaskType>(1);
        y_data[i] = x_data[i];
      }
    }
  }
};

template <typename T>
using CPUDropoutKernel =
    DropoutKernel<platform::CPUDeviceContext, T, CPUDropoutFunctor>;

}  // namespace operators
}  // namespace paddle
