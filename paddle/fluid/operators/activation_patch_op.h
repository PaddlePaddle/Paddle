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
#include <glog/logging.h>
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cmath>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/float16.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
__global__ void KeRelu(const T* x, const int num, T* y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    y[i] = max(x[i], static_cast<T>(0.));
  }
}

template <typename DeviceContext, typename T>
struct ReluFunctor {
  void operator()(const DeviceContext& dev_ctx, const T* x, int num,
                  T* out) const {
    int block = 512;
    int grid = (num + block - 1) / block;
    KeRelu<T><<<grid, block, 0, dev_ctx.stream()>>>(x, num, y);
  }
};

template <typename DeviceContext, typename T, typename Functor>
class ReluCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_t = ctx.Input<Tensor>("X");
    auto* out_t = ctx.Output<Tensor>("Y");
    auto x = in_t->data<T>();
    auto y = out_t->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    Functor<DeviceContext, T> func;
    func(dev_ctx, x, num, y);
  }
};

template <typename DeviceContext, typename T>
__global__ void KeReluGrad(const T* y, const T* dy, const int num, T* dx) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    dx[i] = dy[i] * (y[i] > 0 ? 1. : 0.);
  }
}

template <typename DeviceContext, typename T>
struct ReluGradFunctor {
  void operator()(const DeviceContext& dev_ctx, const T* y, const T* dy,
                  int num, T* dx) const {
    int block = 512;
    int grid = (num + block - 1) / block;
    KeReluGrad<T><<<grid, block, 0, dev_ctx.stream()>>>(y, dy, num, dx);
  }
};

template <typename DeviceContext, typename T, typename Functor>
class ReluGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dy_t = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* y_t = ctx.Input<Tensor>("Y");
    auto* dx_t = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto dy = dy_t->data<T>();
    auto y = y_t->data<T>();
    auto dx = dx_t->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    Functor func;
    func(dev_ctx, y, dy, y->numel(), dx)
  }
};
}  // namespace operators
}  // namespace paddle
