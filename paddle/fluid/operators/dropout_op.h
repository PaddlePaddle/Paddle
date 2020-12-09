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

#include <cstring>
#include <random>
#include <string>

#include <algorithm>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

#ifdef __NVCC__
/*
__global__ void DropoutGradCUDAKernel(const half* dout, const uint8_t* mask,
                                      float factor, const int64_t size,
                                      half* dx) {
  half2 h2_factor = __float2half2_rn(factor);
  const auto* dout_h2 = reinterpret_cast<const half2*>(dout);
  const auto* mask_c2 = reinterpret_cast<const char2*>(mask);
  auto* dx_h2 = reinterpret_cast<half2*>(dx);
  CUDA_KERNEL_LOOP(index, size / 2) {
    char2 mask_val = mask_c2[index];
    half2 mask_h2;
    mask_h2.x = mask_val.x;
    mask_h2.y = mask_val.y;
    dx_h2[index] = __hmul2(__hmul2(dout_h2[index], mask_h2), h2_factor);
  }
  if (index == size / 2 && size % 2 == 1) {
    const int64_t last_idx = size - 1;
    half mask = mask[last_idx];
    dx[last_idx] = __hmul(__hmul(dout[last_idx], mask), h2_factor.x);
  }
}
*/

__global__ void DropoutGradCUDAKernel(const half2* dout, const uint8_t* mask,
                                      const half factor, const int64_t size,
                                      half2* dx) {
  half2 factor_h2 = __half2half2(factor);
  const auto* mask_uc2 = reinterpret_cast<const uchar2*>(mask);
  CUDA_KERNEL_LOOP(index, size / 2) {
    uchar2 mask_val = mask_uc2[index];
    half2 mask_h2;
    mask_h2.x = mask_val.x;
    mask_h2.y = mask_val.y;
    dx[index] = __hmul2(__hmul2(dout[index], mask_h2), factor_h2);
  }
  if (size % 2 != 0 && blockIdx.x == 0 && threadIdx.x == 0) {
    const int64_t last_idx = (size / 2) + 1;
    half mask_h = mask[size - 1];
    dx[last_idx].x = __hmul(__hmul(dout[last_idx].x, mask_h), factor_h2.x);
  }
}
#endif

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class CPUDropoutKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* seed =
        context.HasInput("Seed") ? context.Input<Tensor>("Seed") : nullptr;
    auto* y = context.Output<Tensor>("Out");
    const auto* x_data = x->data<T>();
    auto* y_data = y->mutable_data<T>(context.GetPlace());
    float dropout_prob = context.Attr<float>("dropout_prob");

    auto& dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    bool upscale_in_train = (dropout_implementation == "upscale_in_train");
    if (!context.Attr<bool>("is_test")) {
      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<uint8_t>(context.GetPlace());
      size_t size = framework::product(mask->dims());

      // Special case when dropout_prob is 1.0
      if (dropout_prob == 1.0f) {
        std::memset(y_data, 0, size * sizeof(*y_data));        // NOLINT
        std::memset(mask_data, 0, size * sizeof(*mask_data));  // NOLINT
        return;
      }
      // std::minstd_rand engine;
      // NOTE: fixed seed should only be used in unittest or for debug.
      // Guarantee to use random seed in training.
      int seed_data = 0;
      if (seed) {
        seed_data = *(seed->data<int>());
      } else {
        seed_data =
            context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : 0;
      }
      auto engine = framework::GetCPURandomEngine(seed_data);

      std::uniform_real_distribution<float> dist(0, 1);

      for (size_t i = 0; i < size; ++i) {
        if (dist(*engine) < dropout_prob) {
          mask_data[i] = 0;
          y_data[i] = 0;
        } else {
          mask_data[i] = 1;
          if (upscale_in_train) {
            y_data[i] = x_data[i] / static_cast<T>(1.0f - dropout_prob);
          } else {
            y_data[i] = x_data[i];
          }
        }
      }
    } else {
      if (upscale_in_train) {
        const auto* X_data = x->data<T>();
        auto* Y_data = y->mutable_data<T>(context.GetPlace());
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int i = 0; i < x->numel(); i++) {
          Y_data[i] = X_data[i];
        }
      } else {
        auto X = EigenMatrix<T>::Reshape(*x, 1);
        auto Y = EigenMatrix<T>::Reshape(*y, 1);
        auto& place =
            *context.template device_context<DeviceContext>().eigen_device();
        Y.device(place) = X * static_cast<T>(1.0f - dropout_prob);
      }
    }
  }
};

template <typename DeviceContext, typename T>
class DropoutGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(!context.Attr<bool>("is_test"), true,
                      platform::errors::PreconditionNotMet(
                          "GradOp is only callable when is_test is false"));

    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* mask = context.Input<Tensor>("Mask");
    grad_x->mutable_data<T>(context.GetPlace());

    auto M = EigenVector<uint8_t>::Flatten(*mask);
    auto dX = EigenVector<T>::Flatten(*grad_x);
    auto dY = EigenVector<T>::Flatten(*grad_y);

    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    auto& dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    if (dropout_implementation == "upscale_in_train") {
      float dropout_prob = context.Attr<float>("dropout_prob");
      if (dropout_prob == 1.0f) {
        dX.device(place) = static_cast<T>(0) * dY;
      } else {
        if (platform::is_gpu_place(context.GetPlace()) &&
            std::is_same<T, platform::float16>::value) {
#ifdef __NVCC__
          auto size = grad_x->numel();
          auto round_size = (size + 1) / 2 * 2;
          auto factor = static_cast<half>(1.0f / (1.0f - dropout_prob));
          auto stream = context.cuda_device_context().stream();
          int threads = 512;
          int grid = (round_size + threads - 1) / threads;
          const auto& dev_ctx = context.cuda_device_context();
          int blocks_per_sm = dev_ctx.GetMaxPhysicalThreadCount() /
                              dev_ctx.GetSMCount() / threads;
          grid = std::min(dev_ctx.GetSMCount() * blocks_per_sm, grid);
          const half2* dout_h2 =
              reinterpret_cast<const half2*>(grad_y->data<T>());
          half2* dx_h2 = reinterpret_cast<half2*>(grad_x->data<T>());
          DropoutGradCUDAKernel<<<grid, threads, 0, stream>>>(
              dout_h2, mask->data<uint8_t>(), factor, size, dx_h2);
#endif
        } else {
          dX.device(place) =
              dY * M.cast<T>() / static_cast<T>(1.0f - dropout_prob);
        }
      }
    } else {
      dX.device(place) = dY * M.cast<T>();
    }
  }
};

}  // namespace operators
}  // namespace paddle
