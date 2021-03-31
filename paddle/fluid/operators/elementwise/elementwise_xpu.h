/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#ifdef PADDLE_WITH_XPU
#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"
#include "xpu/refactor/math.h"

namespace paddle {
namespace operators {

static std::pair<std::vector<int>, std::vector<int>> XPUDimsToBroadcastVector(
    const framework::DDim& x, const framework::DDim& y) {
  std::vector<int> x_v;
  std::vector<int> y_v;
  int y_size = y.size();
  for (int i = 0; i < y_size; ++i) {
    if (x[i] == y[i]) {
      x_v.push_back(y[i]);
      y_v.push_back(y[i]);
      continue;
    }
    x_v.push_back(1);
    x_v.push_back(x[i]);
    y_v.push_back(y[i] / x[i]);
    y_v.push_back(x[i]);
  }
  return std::make_pair(x_v, y_v);
}

static std::pair<std::vector<int>, std::vector<int>> XPUReducesAxisVector(
    const framework::DDim& x, const framework::DDim& y) {
  std::vector<int> x_vector;
  std::vector<int> axis_v;
  PADDLE_ENFORCE_GT(
      x.size(), 0, platform::errors::OutOfRange("x size is less 1, x shape is ",
                                                x.to_str()));
  PADDLE_ENFORCE_GT(
      y.size(), 0, platform::errors::OutOfRange("y size is less 1, y shape is ",
                                                y.to_str()));

  int y_nums = framework::product(y);
  x_vector = framework::vectorize<int>(x);
  if (y_nums == 1) {
    for (int i = 0; i < x.size(); ++i) {
      axis_v.push_back(i);
    }
    return std::make_pair(x_vector, axis_v);
  }
  int yidx = 0;
  for (size_t i = 0; i < x_vector.size(); ++i) {
    if (yidx >= y.size() || y[yidx] == 1) {
      axis_v.push_back(i);
      yidx++;
      continue;
    }
    if (x_vector[i] != y[yidx]) {
      axis_v.push_back(i);
      continue;
    }
    yidx++;
  }
  return std::make_pair(x_vector, axis_v);
}

template <typename T>
void XPUElementwise(
    const framework::ExecutionContext& ctx,
    std::function<int(xpu::Context*, const T*, const T*, T*, int)> func) {
  auto x_var = ctx.InputVar("X");
  PADDLE_ENFORCE_NE(x_var, nullptr, platform::errors::InvalidArgument(
                                        "Cannot get input Variable X"));
  PADDLE_ENFORCE_EQ(
      x_var->IsType<framework::LoDTensor>(), true,
      platform::errors::InvalidArgument(
          "XPU only support LoDTensor, Input(X) is not LoDTensor"));

  auto x = x_var->Get<framework::LoDTensor>();
  auto* y = ctx.Input<framework::LoDTensor>("Y");
  auto* z = ctx.Output<framework::LoDTensor>("Out");
  z->mutable_data<T>(ctx.GetPlace());
  auto x_dims = x.dims();
  auto y_dims = y->dims();
  int max_dim = std::max(x_dims.size(), y_dims.size());
  int axis = ctx.Attr<int>("axis");
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);

  PADDLE_ENFORCE_GE(
      axis, 0,
      platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis, max_dim,
                    platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim, axis));

  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array.data(),
                         y_dims_array.data(), out_dims_array.data(), max_dim,
                         axis);
  framework::DDim out_dim = framework::make_ddim(out_dims_array);

  const T* x_data = x.data<T>();
  const T* y_data = y->data<T>();
  T* z_data = z->data<T>();
  bool need_wait = false;
  framework::Tensor x_broadcast_tensor;
  framework::Tensor y_broadcast_tensor;
  auto& dev_ctx =
      ctx.template device_context<paddle::platform::XPUDeviceContext>();
  int ret = xpu::SUCCESS;
  // begin broadcast now
  if (x.numel() != z->numel()) {
    // broadcast x
    std::pair<std::vector<int>, std::vector<int>> bcast_v =
        XPUDimsToBroadcastVector(framework::make_ddim(x_dims_array), out_dim);

    ret = xpu::broadcast<T>(dev_ctx.x_context(), x_data,
                            x_broadcast_tensor.mutable_data<T>(
                                ctx.GetPlace(), z->numel() * sizeof(T)),
                            bcast_v.first, bcast_v.second);
    PADDLE_ENFORCE_EQ(
        ret, xpu::SUCCESS,
        platform::errors::External(
            "XPU kernel broadcast occur error in XPUElementwise error code %d",
            ret));
    need_wait = true;
    x_data = x_broadcast_tensor.data<T>();
  }

  if (y->numel() != z->numel()) {
    // broadcast y
    std::vector<int> bcast_x_v;
    std::vector<int> bcast_y_v;
    std::pair<std::vector<int>, std::vector<int>> bcast_v =
        XPUDimsToBroadcastVector(framework::make_ddim(y_dims_array), out_dim);
    ret = xpu::broadcast<T>(dev_ctx.x_context(), y_data,
                            y_broadcast_tensor.mutable_data<T>(
                                ctx.GetPlace(), z->numel() * sizeof(T)),
                            bcast_v.first, bcast_v.second);
    PADDLE_ENFORCE_EQ(
        ret, xpu::SUCCESS,
        platform::errors::External(
            "XPU kernel broadcast occur error in XPUElementwise error code %d",
            ret));
    need_wait = true;
    y_data = y_broadcast_tensor.data<T>();
  }
  int len = z->numel();
  ret = func(dev_ctx.x_context(), x_data, y_data, z_data, len);
  PADDLE_ENFORCE_EQ(
      ret, xpu::SUCCESS,
      platform::errors::External(
          "XPU kernel Elementwise occur error in XPUElementwise error code ",
          ret));

  if (need_wait && dev_ctx.x_context()->xpu_stream) {
    dev_ctx.Wait();
  }
}

template <typename T>
void XPUElementwiseGrad(const framework::ExecutionContext& ctx,
                        std::function<int(xpu::Context*, const T*, const T*,
                                          const T*, const T*, T*, T*, int len)>
                            func,
                        bool use_x_y_data) {
  auto* x = ctx.Input<framework::Tensor>("X");
  auto* y = ctx.Input<framework::Tensor>("Y");
  auto* dz = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
  auto* z = dz;
  auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
  auto* dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
  int axis = ctx.Attr<int>("axis");
  const framework::DDim& x_dims = x->dims();
  const framework::DDim& y_dims = y->dims();
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis, 0,
      platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis, max_dim,
                    platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim, axis));

  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array.data(),
                         y_dims_array.data(), out_dims_array.data(), max_dim,
                         axis);
  framework::DDim out_dim = framework::make_ddim(out_dims_array);

  int len = framework::product(out_dim);

  framework::Tensor x_broadcast_tensor;
  framework::Tensor y_broadcast_tensor;

  framework::Tensor dx_local_tensor;
  framework::Tensor dy_local_tensor;

  bool need_wait = false;
  const T* x_data = use_x_y_data ? x->data<T>() : z->data<T>();
  const T* y_data = use_x_y_data ? y->data<T>() : z->data<T>();

  const T* z_data = z->data<T>();
  const T* dz_data = (const T*)dz->data<T>();

  bool dx_need_reduce = (dx != nullptr) && (dx->numel() != len);
  bool dy_need_reduce = (dy != nullptr) && (dy->numel() != len);

  T* dx_data =
      ((dx == nullptr) || dx_need_reduce)
          ? (dx_local_tensor.mutable_data<T>(ctx.GetPlace(), len * sizeof(T)))
          : (dx->mutable_data<T>(ctx.GetPlace()));

  T* dy_data =
      ((dy == nullptr) || dy_need_reduce)
          ? (dy_local_tensor.mutable_data<T>(ctx.GetPlace(), len * sizeof(T)))
          : (dy->mutable_data<T>(ctx.GetPlace()));

  int ret = xpu::SUCCESS;
  auto& dev_ctx =
      ctx.template device_context<paddle::platform::XPUDeviceContext>();

  if (use_x_y_data && x->numel() != len) {
    std::vector<int> bcast_x_v;
    std::vector<int> bcast_y_v;
    std::pair<std::vector<int>, std::vector<int>> bcast_v =
        XPUDimsToBroadcastVector(framework::make_ddim(x_dims_array), out_dim);
    ret = xpu::broadcast<T>(
        dev_ctx.x_context(), x_data,
        x_broadcast_tensor.mutable_data<T>(ctx.GetPlace(), len * sizeof(T)),
        bcast_v.first, bcast_v.second);
    PADDLE_ENFORCE_EQ(ret, xpu::SUCCESS,
                      platform::errors::External(
                          "XPU kernel broadcast error occur! %d", ret));
    need_wait = true;
    x_data = x_broadcast_tensor.data<T>();
  }

  if (use_x_y_data && y->numel() != len) {
    // broadcast y
    std::vector<int> bcast_x_v;
    std::vector<int> bcast_y_v;
    std::pair<std::vector<int>, std::vector<int>> bcast_v =
        XPUDimsToBroadcastVector(framework::make_ddim(y_dims_array), out_dim);
    ret = xpu::broadcast<T>(
        dev_ctx.x_context(), y_data,
        y_broadcast_tensor.mutable_data<T>(ctx.GetPlace(), len * sizeof(T)),
        bcast_v.first, bcast_v.second);
    PADDLE_ENFORCE_EQ(ret, xpu::SUCCESS,
                      platform::errors::External(
                          "XPU kernel broadcast error occur! %d", ret));
    need_wait = true;
    y_data = y_broadcast_tensor.data<T>();
  }

  ret = func(dev_ctx.x_context(), x_data, y_data, z_data, dz_data, dx_data,
             dy_data, len);
  PADDLE_ENFORCE_EQ(ret, xpu::SUCCESS, platform::errors::External(
                                           "XPU kernel binary occur error in "
                                           "XPUElementwiseGrad, error code %d",
                                           ret));

  if (dx_need_reduce) {
    const framework::DDim& dx_dims = dx->dims();
    std::pair<std::vector<int>, std::vector<int>> reduce_v =
        XPUReducesAxisVector(out_dim, dx_dims);
    ret = xpu::reduce_sum<T>(dev_ctx.x_context(), dx_data,
                             dx->mutable_data<T>(ctx.GetPlace()),
                             reduce_v.first, reduce_v.second);
    PADDLE_ENFORCE_EQ(
        ret, xpu::SUCCESS,
        platform::errors::External("XPU kernel reduce_sum occur error in "
                                   "XPUElementwiseGrad, error code %d",
                                   ret));
    need_wait = true;
  }

  if (dy_need_reduce) {
    const framework::DDim& dy_dims = dy->dims();
    std::pair<std::vector<int>, std::vector<int>> reduce_v =
        XPUReducesAxisVector(out_dim, dy_dims);
    ret = xpu::reduce_sum<T>(dev_ctx.x_context(), dy_data,
                             dy->mutable_data<T>(ctx.GetPlace()),
                             reduce_v.first, reduce_v.second);
    PADDLE_ENFORCE_EQ(
        ret, xpu::SUCCESS,
        platform::errors::External("XPU kernel reduce_sum occur error in "
                                   "XPUElementwiseGrad, error code %d",
                                   ret));
    need_wait = true;
  }

  if (need_wait && dev_ctx.x_context()->xpu_stream) {
    dev_ctx.Wait();
  }
}

}  // namespace operators
}  // namespace paddle
#endif
