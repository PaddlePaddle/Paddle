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

#include <memory>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

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

template <typename DeviceContext, typename VecDxType, typename T>
static inline void StackGradFunctorForRange(const DeviceContext &ctx,
                                            const VecDxType &dx, const T *dy,
                                            int total_num, int n, int post) {
  platform::ForRange<DeviceContext> for_range(ctx, total_num);
  for_range(StackGradFunctor<VecDxType, T>(dx, dy, n, post));
}

template <typename DeviceContext, typename T>
class StackKernel : public framework::OpKernel<T> {
  using Tensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    // Determine whether the element of X is LoDTensorArray.
    const std::vector<framework::Variable *> &x_vars = ctx.MultiInputVar("X");
    bool is_tensor_array = x_vars[0]->IsType<framework::LoDTensorArray>();

    auto *y = ctx.Output<Tensor>("Y");
    int axis = ctx.Attr<int>("axis");
    std::vector<const Tensor *> x;
    framework::LoDTensorArray x_array;

    if (is_tensor_array) {
      auto x_array_list = ctx.MultiInput<framework::LoDTensorArray>("X");
      x_array = x_array_list[0][0];  // a vector of LoDTensor
      if (axis < 0) axis += (x_array[0].dims().size() + 1);
    } else {
      x = ctx.MultiInput<Tensor>("X");
      if (axis < 0) axis += (x[0]->dims().size() + 1);
    }

    int n = is_tensor_array ? static_cast<int>(x_array.size())
                            : static_cast<int>(x.size());
    framework::DDim dim = is_tensor_array ? x_array[0].dims() : x[0]->dims();
    std::vector<const T *> x_datas(n);
    for (int i = 0; i < n; i++) {
      x_datas[i] = is_tensor_array ? x_array[i].data<T>() : x[i]->data<T>();
    }

    auto vec = framework::vectorize<int>(dim);
    int pre = 1, post = 1;
    for (auto i = 0; i < axis; ++i) pre *= dim[i];
    for (auto i = axis; i < dim.size(); ++i) post *= dim[i];
    vec.insert(vec.begin() + axis, n);
    y->Resize(framework::make_ddim(vec));
    auto *y_data = y->mutable_data<T>(ctx.GetPlace());

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
  }
};

template <typename DeviceContext, typename T>
class StackGradKernel : public framework::OpKernel<T> {
  using Tensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto dx = ctx.MultiOutput<Tensor>(framework::GradVarName("X"));
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis += dy->dims().size();

    int n = dy->dims()[axis];
    std::vector<T *> dx_datas(n);  // NOLINT
    for (int i = 0; i < n; i++) {
      dx_datas[i] = dx[i]->mutable_data<T>(ctx.GetPlace());
    }
    auto dy_data = dy->data<T>();

    int pre = 1;
    for (int i = 0; i < axis; ++i) pre *= dy->dims()[i];
    int total_num = dy->numel();
    int post = total_num / (n * pre);

    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    auto dx_data_arr = dx_datas.data();
    StackGradFunctorForRange(dev_ctx, dx_data_arr, dy_data, total_num, n, post);
  }
};

}  // namespace operators
}  // namespace paddle
