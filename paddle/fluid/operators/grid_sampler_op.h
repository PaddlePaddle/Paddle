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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

using Array3 = Eigen::DSizes<int64_t, 3>;
using Array4 = Eigen::DSizes<int64_t, 4>;

template <typename T>
static inline bool isInBound(T x, T y, T x_max, T y_max) {
  if (x < 0 || x > x_max || y < 0 || y > y_max) {
    return false;
  }
  return true;
}

template <typename T>
static void CalcGridLocations(const platform::CPUDeviceContext& ctx,
                              const Tensor& grid, Tensor* x_w, Tensor* x_e,
                              Tensor* y_n, Tensor* y_s, Tensor* d_w,
                              Tensor* d_e, Tensor* d_n, Tensor* d_s) {
  auto& place = *ctx.eigen_device();
  const int n = grid.dims()[0];
  const int h = grid.dims()[1];
  const int w = grid.dims()[2];
  const T x_max = static_cast<T>(w - 1);
  const T y_max = static_cast<T>(h - 1);

  // split grid with shape (n, h, w, 2) into (x, y) by the 3rd Dim
  Tensor grid_x, grid_y;
  T* grid_x_data = grid_x.mutable_data<T>({n, h, w}, ctx.GetPlace());
  T* grid_y_data = grid_y.mutable_data<T>({n, h, w}, ctx.GetPlace());
  const T* grid_data = grid.data<T>();
  for (int i = 0; i < n * h * w; i++) {
    grid_x_data[i] = grid_data[2 * i];
    grid_y_data[i] = grid_data[(2 * i) + 1];
  }

  Tensor ones;
  ones.mutable_data<T>({n, h, w}, ctx.GetPlace());
  auto ones_t = EigenTensor<T, 3>::From(ones).setConstant(1.0);
  Tensor half_xmax;
  Tensor half_ymax;
  half_xmax.mutable_data<T>({n, h, w}, ctx.GetPlace());
  auto half_xmax_t =
      EigenTensor<T, 3>::From(half_xmax).setConstant(0.5 * x_max);
  half_ymax.mutable_data<T>({n, h, w}, ctx.GetPlace());
  auto half_ymax_t =
      EigenTensor<T, 3>::From(half_ymax).setConstant(0.5 * y_max);

  // scale grid to [0, h-1/w-1]
  auto grid_x_t = EigenTensor<T, 3>::From(grid_x);
  auto grid_y_t = EigenTensor<T, 3>::From(grid_y);
  grid_x_t.device(place) = (grid_x_t + ones_t) * half_xmax_t;
  grid_y_t.device(place) = (grid_y_t + ones_t) * half_ymax_t;

  // calculate coords of 4 corner points
  x_w->mutable_data<T>({n, h, w}, ctx.GetPlace());
  x_e->mutable_data<T>({n, h, w}, ctx.GetPlace());
  y_n->mutable_data<T>({n, h, w}, ctx.GetPlace());
  y_s->mutable_data<T>({n, h, w}, ctx.GetPlace());
  auto x_w_t = EigenTensor<T, 3>::From(*x_w);
  auto x_e_t = EigenTensor<T, 3>::From(*x_e);
  auto y_n_t = EigenTensor<T, 3>::From(*y_n);
  auto y_s_t = EigenTensor<T, 3>::From(*y_s);
  x_w_t.device(place) = grid_x_t.floor();
  x_e_t.device(place) = x_w_t + ones_t;
  y_n_t.device(place) = grid_y_t.floor();
  y_s_t.device(place) = y_n_t + ones_t;

  // calculate distances to 4 sides
  d_w->mutable_data<T>({n, h, w}, ctx.GetPlace());
  d_e->mutable_data<T>({n, h, w}, ctx.GetPlace());
  d_n->mutable_data<T>({n, h, w}, ctx.GetPlace());
  d_s->mutable_data<T>({n, h, w}, ctx.GetPlace());
  auto d_w_t = EigenTensor<T, 3>::From(*d_w);
  auto d_e_t = EigenTensor<T, 3>::From(*d_e);
  auto d_n_t = EigenTensor<T, 3>::From(*d_n);
  auto d_s_t = EigenTensor<T, 3>::From(*d_s);
  d_w_t.device(place) = grid_x_t - x_w_t;
  d_e_t.device(place) = x_e_t - grid_x_t;
  d_n_t.device(place) = grid_y_t - y_n_t;
  d_s_t.device(place) = y_s_t - grid_y_t;
}

template <typename T>
static void GetGridPointValue(const Tensor& input, Tensor* output,
                              const Tensor& x, const Tensor& y) {
  const int n = input.dims()[0];
  const int c = input.dims()[1];
  const int h = input.dims()[2];
  const int w = input.dims()[3];
  auto x_t = EigenTensor<T, 3>::From(x);
  auto y_t = EigenTensor<T, 3>::From(y);
  auto output_t = EigenTensor<T, 4>::From(*output).setConstant((T)0);
  auto input_t = EigenTensor<T, 4>::From(input);

  for (int i = 0; i < n; i++) {
    for (int k = 0; k < h; k++) {
      for (int l = 0; l < w; l++) {
        if (isInBound(x_t(i, k, l), y_t(i, k, l), (T)(w - 1), (T)(h - 1))) {
          for (int j = 0; j < c; j++) {
            output_t(i, j, k, l) =
                input_t(i, j, static_cast<int>(round(y_t(i, k, l))),
                        static_cast<int>(round(x_t(i, k, l))));
          }
        }
      }
    }
  }
}

template <typename T>
static void GatherOutputGradToInputGrad(const Tensor& output_grad,
                                        Tensor* input_grad, const Tensor& x,
                                        const Tensor& y, const Tensor& d1,
                                        const Tensor& d2) {
  const int n = output_grad.dims()[0];
  const int c = output_grad.dims()[1];
  const int h = output_grad.dims()[2];
  const int w = output_grad.dims()[3];
  auto x_t = EigenTensor<T, 3>::From(x);
  auto y_t = EigenTensor<T, 3>::From(y);
  auto d1_t = EigenTensor<T, 3>::From(d1);
  auto d2_t = EigenTensor<T, 3>::From(d2);
  auto input_grad_t = EigenTensor<T, 4>::From(*input_grad);
  auto output_grad_t = EigenTensor<T, 4>::From(output_grad);

  for (int i = 0; i < n; i++) {
    for (int k = 0; k < h; k++) {
      for (int l = 0; l < w; l++) {
        if (isInBound(x_t(i, k, l), y_t(i, k, l), (T)(w - 1), (T)(h - 1))) {
          for (int j = 0; j < c; j++) {
            input_grad_t(i, j, static_cast<int>(round(y_t(i, k, l))),
                         static_cast<int>(round(x_t(i, k, l)))) +=
                output_grad_t(i, j, k, l) * d1_t(i, k, l) * d2_t(i, k, l);
          }
        }
      }
    }
  }
}

template <typename DeviceContext, typename T>
class GridSampleOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto* input = ctx.Input<Tensor>("X");
    auto* grid = ctx.Input<Tensor>("Grid");

    const int n = input->dims()[0];
    const int c = input->dims()[1];
    const int h = input->dims()[2];
    const int w = input->dims()[3];

    // calc locations and distances of 4 corner points
    Tensor x_w, x_e, y_n, y_s;
    Tensor d_w, d_e, d_n, d_s;
    CalcGridLocations<T>(
        ctx.template device_context<platform::CPUDeviceContext>(), *grid, &x_w,
        &x_e, &y_n, &y_s, &d_w, &d_e, &d_n, &d_s);

    auto* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>({n, c, h, w}, ctx.GetPlace());
    math::SetConstant<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), output,
        static_cast<T>(0));

    // calc 4 corner points value
    Tensor v_wn, v_en, v_ws, v_es;
    v_wn.mutable_data<T>({n, c, h, w}, ctx.GetPlace());
    v_en.mutable_data<T>({n, c, h, w}, ctx.GetPlace());
    v_ws.mutable_data<T>({n, c, h, w}, ctx.GetPlace());
    v_es.mutable_data<T>({n, c, h, w}, ctx.GetPlace());
    GetGridPointValue<T>(*input, &v_wn, x_w, y_n);
    GetGridPointValue<T>(*input, &v_en, x_e, y_n);
    GetGridPointValue<T>(*input, &v_ws, x_w, y_s);
    GetGridPointValue<T>(*input, &v_es, x_e, y_s);

    auto d_w_t = EigenTensor<T, 3>::From(d_w);
    auto d_e_t = EigenTensor<T, 3>::From(d_e);
    auto d_n_t = EigenTensor<T, 3>::From(d_n);
    auto d_s_t = EigenTensor<T, 3>::From(d_s);
    auto d_w_scaled_t =
        d_w_t.reshape(Array4(n, 1, h, w)).broadcast(Array4(1, c, 1, 1));
    auto d_e_scaled_t =
        d_e_t.reshape(Array4(n, 1, h, w)).broadcast(Array4(1, c, 1, 1));
    auto d_n_scaled_t =
        d_n_t.reshape(Array4(n, 1, h, w)).broadcast(Array4(1, c, 1, 1));
    auto d_s_scaled_t =
        d_s_t.reshape(Array4(n, 1, h, w)).broadcast(Array4(1, c, 1, 1));
    auto v_wn_t = EigenTensor<T, 4>::From(v_wn);
    auto v_en_t = EigenTensor<T, 4>::From(v_en);
    auto v_ws_t = EigenTensor<T, 4>::From(v_ws);
    auto v_es_t = EigenTensor<T, 4>::From(v_es);
    auto output_t = EigenTensor<T, 4>::From(*output);
    // bilinear interpolaetion by 4 corner points
    output_t.device(place) = v_wn_t * d_e_scaled_t * d_s_scaled_t +
                             v_en_t * d_w_scaled_t * d_s_scaled_t +
                             v_ws_t * d_e_scaled_t * d_n_scaled_t +
                             v_es_t * d_w_scaled_t * d_n_scaled_t;
  }
};

template <typename DeviceContext, typename T>
class GridSampleGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* grid = ctx.Input<Tensor>("Grid");
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));

    const int n = input->dims()[0];
    const int c = input->dims()[1];
    const int h = input->dims()[2];
    const int w = input->dims()[3];

    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    input_grad->mutable_data<T>({n, c, h, w}, ctx.GetPlace());
    math::SetConstant<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), input_grad,
        static_cast<T>(0));
    auto* grid_grad = ctx.Output<Tensor>(framework::GradVarName("Grid"));
    grid_grad->mutable_data<T>({n, h, w, 2}, ctx.GetPlace());
    math::SetConstant<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), grid_grad,
        static_cast<T>(0));

    Tensor x_w, x_e, y_n, y_s;
    Tensor d_w, d_e, d_n, d_s;
    CalcGridLocations<T>(
        ctx.template device_context<platform::CPUDeviceContext>(), *grid, &x_w,
        &x_e, &y_n, &y_s, &d_w, &d_e, &d_n, &d_s);

    // gather output grad value to input grad by corner point coords and weight
    GatherOutputGradToInputGrad<T>(*output_grad, input_grad, x_w, y_n, d_e,
                                   d_s);
    GatherOutputGradToInputGrad<T>(*output_grad, input_grad, x_w, y_s, d_e,
                                   d_n);
    GatherOutputGradToInputGrad<T>(*output_grad, input_grad, x_e, y_n, d_w,
                                   d_s);
    GatherOutputGradToInputGrad<T>(*output_grad, input_grad, x_e, y_s, d_w,
                                   d_n);

    // calc 4 corner points value
    Tensor v_wn, v_en, v_ws, v_es;
    v_wn.mutable_data<T>({n, c, h, w}, ctx.GetPlace());
    v_en.mutable_data<T>({n, c, h, w}, ctx.GetPlace());
    v_ws.mutable_data<T>({n, c, h, w}, ctx.GetPlace());
    v_es.mutable_data<T>({n, c, h, w}, ctx.GetPlace());
    GetGridPointValue<T>(*input, &v_wn, x_w, y_n);
    GetGridPointValue<T>(*input, &v_en, x_e, y_n);
    GetGridPointValue<T>(*input, &v_ws, x_w, y_s);
    GetGridPointValue<T>(*input, &v_es, x_e, y_s);
    auto v_wn_t = EigenTensor<T, 4>::From(v_wn);
    auto v_en_t = EigenTensor<T, 4>::From(v_en);
    auto v_ws_t = EigenTensor<T, 4>::From(v_ws);
    auto v_es_t = EigenTensor<T, 4>::From(v_es);

    auto d_w_t = EigenTensor<T, 3>::From(d_w);
    auto d_e_t = EigenTensor<T, 3>::From(d_e);
    auto d_n_t = EigenTensor<T, 3>::From(d_n);
    auto d_s_t = EigenTensor<T, 3>::From(d_s);

    auto output_grad_t = EigenTensor<T, 4>::From(*output_grad);

    Tensor grid_grad_x, grid_grad_y;
    grid_grad_x.mutable_data<T>({n, h, w}, ctx.GetPlace());
    grid_grad_y.mutable_data<T>({n, h, w}, ctx.GetPlace());
    auto grid_grad_x_t = EigenTensor<T, 3>::From(grid_grad_x).setConstant(0.0);
    auto grid_grad_y_t = EigenTensor<T, 3>::From(grid_grad_y).setConstant(0.0);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < c; j++) {
        for (int k = 0; k < h; k++) {
          for (int l = 0; l < w; l++) {
            grid_grad_x_t(i, k, l) +=
                ((v_en_t(i, j, k, l) - v_wn_t(i, j, k, l)) * d_s_t(i, k, l) +
                 (v_es_t(i, j, k, l) - v_ws_t(i, j, k, l)) * d_n_t(i, k, l)) *
                output_grad_t(i, j, k, l);
            grid_grad_y_t(i, k, l) +=
                ((v_ws_t(i, j, k, l) - v_wn_t(i, j, k, l)) * d_e_t(i, k, l) +
                 (v_es_t(i, j, k, l) - v_en_t(i, j, k, l)) * d_w_t(i, k, l)) *
                output_grad_t(i, j, k, l);
          }
        }
      }
    }
    const T x_max = static_cast<T>(w - 1);
    const T y_max = static_cast<T>(h - 1);
    grid_grad_x_t = grid_grad_x_t * (x_max / (T)2);
    grid_grad_y_t = grid_grad_y_t * (y_max / (T)2);

    // gather grid_grad [x, y] in 3rd Dim
    T* grid_grad_data = grid_grad->data<T>();
    T* grid_grad_x_data = grid_grad_x.data<T>();
    T* grid_grad_y_data = grid_grad_y.data<T>();
    for (int i = 0; i < n * h * w; i++) {
      grid_grad_data[2 * i] = grid_grad_x_data[i];
      grid_grad_data[2 * i + 1] = grid_grad_y_data[i];
    }
  }
};

}  // namespace operators
}  // namespace paddle
