// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <random>

#include "gtest/gtest.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
static void InitRandom(phi::DenseTensor *tensor, const platform::Place &place) {
  phi::DenseTensor cpu_tensor;
  auto *cpu_ptr =
      cpu_tensor.mutable_data<T>(tensor->dims(), platform::CPUPlace());
  int64_t numel = cpu_tensor.numel();
  std::mt19937 engine;
  std::uniform_real_distribution<T> dist(static_cast<T>(-2.0),
                                         static_cast<T>(2.0));
  for (int64_t i = 0; i < numel; ++i) {
    cpu_ptr[i] = dist(engine);
  }
  framework::TensorCopySync(cpu_tensor, place, tensor);
}

template <typename T>
struct LeakyReluGradGradEachElementFunctor {
  LeakyReluGradGradEachElementFunctor(const T *ddx,
                                      const T *x,
                                      T alpha,
                                      T *ddout)
      : ddx_(ddx), x_(x), alpha_(alpha), ddout_(ddout) {}

  HOSTDEVICE void operator()(int idx) {
    if (x_[idx] >= 0) {
      ddout_[idx] = ddx_[idx];
    } else {
      ddout_[idx] = ddx_[idx] * alpha_;
    }
  }

  const T *ddx_;
  const T *x_;
  T alpha_;
  T *ddout_;
};

template <typename T>
static bool TestLeakyReluGradGradMain(const framework::DDim &dim,
                                      const platform::Place &place,
                                      float alpha) {
  LeakyReluGradGradFunctor<T> functor;
  functor.alpha = alpha;
  auto &dev_ctx = *platform::DeviceContextPool::Instance().Get(place);
  phi::DenseTensor *out = nullptr;
  phi::DenseTensor *dout = nullptr;
  phi::DenseTensor *dx = nullptr;

  phi::DenseTensor x;
  x.Resize(dim);
  InitRandom<T>(&x, place);

  phi::DenseTensor ddx;
  ddx.Resize(dim);
  InitRandom<T>(&ddx, place);

  phi::DenseTensor ddout;
  ddout.Resize(dim);
  InitRandom<T>(&ddout, place);

  phi::DenseTensor ddout_actual;
  ddout_actual.mutable_data<T>(dim, place);
  LeakyReluGradGradEachElementFunctor<T> actual_functor(ddx.data<T>(),
                                                        x.data<T>(),
                                                        static_cast<T>(alpha),
                                                        ddout_actual.data<T>());

  int64_t limit = x.numel();

#if defined(__NVCC__) || defined(__HIPCC__)
  if (platform::is_gpu_place(place)) {
    auto &cuda_dev_ctx = dynamic_cast<phi::GPUContext &>(dev_ctx);
    functor(cuda_dev_ctx, &x, out, &ddx, &ddout, dout, dx);
    platform::ForRange<phi::GPUContext> for_range(cuda_dev_ctx, limit);
    for_range(actual_functor);
  } else {
#endif
    auto &cpu_dev_ctx = dynamic_cast<phi::CPUContext &>(dev_ctx);
    functor(cpu_dev_ctx, &x, out, &ddx, &ddout, dout, dx);
    platform::ForRange<phi::CPUContext> for_range(cpu_dev_ctx, limit);
    for_range(actual_functor);
#if defined(__NVCC__) || defined(__HIPCC__)
  }
#endif

  dev_ctx.Wait();

  phi::DenseTensor ddout_cpu, ddout_actual_cpu;
  framework::TensorCopySync(ddout, platform::CPUPlace(), &ddout_cpu);
  framework::TensorCopySync(
      ddout_actual, platform::CPUPlace(), &ddout_actual_cpu);

  bool is_equal = std::equal(ddout_cpu.data<T>(),
                             ddout_cpu.data<T>() + limit,
                             ddout_actual_cpu.data<T>());
  return is_equal;
}

}  // namespace operators
}  // namespace paddle
