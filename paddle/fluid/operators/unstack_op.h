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

#include <memory>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include <thrust/device_vector.h>
#endif

namespace paddle {
namespace operators {

template <typename VecXType, typename T>
struct StackFunctor {
  HOSTDEVICE StackFunctor(const VecXType &x, T *y, int n, int post)
      : x_(x), y_(y), n_(n), post_(post) {}

  HOSTDEVICE void operator()(int idx) {
    int i = idx / (n_ * post_);
    int which_x = idx / post_ - i * n_;
    int x_index = i * post_ + idx % post_;
    y_[idx] = x_[which_x][x_index];
  }

 private:
  VecXType x_;
  T *y_;
  int n_;
  int post_;
};

template <typename VecDxType, typename T>
struct StackGradFunctor {
  HOSTDEVICE StackGradFunctor(const VecDxType &dx, const T *dy, int n, int post)
      : dx_(dx), dy_(dy), n_(n), post_(post) {}

  HOSTDEVICE void operator()(int idx) {
    int i = idx / (n_ * post_);
    int which_x = idx / post_ - i * n_;
    int x_index = i * post_ + idx % post_;
    dx_[which_x][x_index] = dy_[idx];
  }

 private:
  VecDxType dx_;
  const T *dy_;
  int n_;
  int post_;
};

template <typename DeviceContext, typename VecXType, typename T>
static inline void StackFunctorForRange(const DeviceContext &ctx,
                                        const VecXType &x, T *y, int total_num,
                                        int n, int post) {
  platform::ForRange<DeviceContext> for_range(ctx, total_num);
  for_range(StackFunctor<VecXType, T>(x, y, n, post));
}

template <typename DeviceContext, typename VecDxType, typename T>
static inline void StackGradFunctorForRange(const DeviceContext &ctx,
                                            const VecDxType &dx, const T *dy,
                                            int total_num, int n, int post) {
  platform::ForRange<DeviceContext> for_range(ctx, total_num);
  for_range(StackGradFunctor<VecDxType, T>(dx, dy, n, post));
}

template <typename DeviceContext, typename T>
class UnStackGradKernel : public framework::OpKernel<T> {
  using Tensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto x = ctx.MultiInput<Tensor>(framework::GradVarName("Y"));
    auto *y = ctx.Output<Tensor>(framework::GradVarName("X"));

    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis += (x[0]->dims().size() + 1);

    int n = static_cast<int>(x.size());
    auto *y_data = y->mutable_data<T>(ctx.GetPlace());
    std::vector<const T *> x_datas(n);
    for (int i = 0; i < n; i++) x_datas[i] = x[i]->data<T>();

    int pre = 1;
    int post = 1;
    auto &dim = x[0]->dims();
    for (auto i = 0; i < axis; ++i) pre *= dim[i];
    for (auto i = axis; i < dim.size(); ++i) post *= dim[i];

#if defined(__NVCC__) || defined(__HIPCC__)
    int total_num = pre * n * post;
    auto &dev_ctx = ctx.template device_context<DeviceContext>();

    thrust::device_vector<const T *> device_x_vec(x_datas);
    auto x_data_arr = device_x_vec.data().get();

    StackFunctorForRange(dev_ctx, x_data_arr, y_data, total_num, n, post);

    // Wait() must be called because device_x_vec may be destructed before
    // kernel ends
    dev_ctx.Wait();
#else
    auto x_data_arr = x_datas.data();

    size_t x_offset = 0;
    size_t y_offset = 0;
    for (int i = 0; i < pre; i++) {
      for (int j = 0; j < n; j++) {
        std::memcpy(y_data + y_offset, x_data_arr[j] + x_offset,
                    post * sizeof(T));
        y_offset += post;
      }
      x_offset += post;
    }
#endif
  }
};

template <typename DeviceContext, typename T>
class UnStackKernel : public framework::OpKernel<T> {
  using Tensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *dy = ctx.Input<Tensor>("X");
    auto dx = ctx.MultiOutput<Tensor>("Y");
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis += dy->dims().size();

    int n = dy->dims()[axis];
    std::vector<T *> dx_datas(n);  // NOLINT
    for (int i = 0; i < n; i++) {
      dx_datas[i] = dx[i]->mutable_data<T>(ctx.GetPlace());
    }
    auto dy_data = dy->data<T>();
    if (dy->numel() == 0) return;
    int pre = 1;
    for (int i = 0; i < axis; ++i) pre *= dy->dims()[i];
    int total_num = dy->numel();
    int post = total_num / (n * pre);

    auto &dev_ctx = ctx.template device_context<DeviceContext>();
#if defined(__NVCC__) || defined(__HIPCC__)
    thrust::device_vector<T *> device_dx_vec(dx_datas);
    auto dx_data_arr = device_dx_vec.data().get();
#else
    auto dx_data_arr = dx_datas.data();
#endif
    StackGradFunctorForRange(dev_ctx, dx_data_arr, dy_data, total_num, n, post);
#if defined(__NVCC__) || defined(__HIPCC__)
    // Wait() must be called because device_dx_vec may be destructed before
    // kernel ends
    dev_ctx.Wait();
#endif
  }
};

}  // namespace operators
}  // namespace paddle
