// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/utils/optional.h"
namespace phi {

enum GRUActivationType { identity = 0, sigmoid = 1, tanh = 2, relu = 3 };

template <typename T, typename Device, typename X, typename Y>
void ActCompute(
    const int act_type, const Device& d, X x, Y y, phi::Place place) {
  if (act_type == identity) {
    y.device(d) = x;
  } else if (act_type == sigmoid) {
    phi::funcs::SigmoidFunctor<T>()(d, x, y);
  } else if (act_type == tanh) {
    phi::funcs::TanhFunctor<T>()(d, x, y);
  } else if (act_type == relu) {
    if (place == phi::CPUPlace())
      phi::funcs::ReluCPUFunctor<T>()(d, x, y);
    else
      phi::funcs::ReluCUDAFunctor<T>()(d, x, y);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported activation type, only supports identity, sigmoid, tanh "
        "and relu."));
  }
}

#define ACT_COMPUTE ActCompute<T>

template <typename T, typename Context>
void GRUUnitKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   const DenseTensor& hidden_prev,
                   const DenseTensor& weight,
                   const paddle::optional<DenseTensor>& bias,
                   int activation,
                   int gate_activation,
                   bool origin_mode,
                   DenseTensor* gate,
                   DenseTensor* reset_hidden_prev,
                   DenseTensor* hidden) {
  auto* input_p = &input;
  auto* hidden_prev_p = &hidden_prev;

  dev_ctx.template Alloc<T>(gate);
  dev_ctx.template Alloc<T>(reset_hidden_prev);
  dev_ctx.template Alloc<T>(hidden);

  int batch_size = input_p->dims()[0];
  int frame_size = hidden_prev_p->dims()[1];

  auto x = phi::EigenMatrix<T>::From(input);
  auto h_p = phi::EigenMatrix<T>::From(hidden_prev);
  auto g = phi::EigenMatrix<T>::From(*gate);
  auto r_h_p = phi::EigenMatrix<T>::From(*reset_hidden_prev);
  auto h = phi::EigenMatrix<T>::From(*hidden);
  auto& place = *dev_ctx.eigen_device();

  // calculate unactivated gate outputs
  if (bias) {
    auto b = phi::EigenMatrix<T>::From(bias.get());
    g.device(place) =
        x + b.reshape(Eigen::array<int, 2>({{1, frame_size * 3}}))
                .broadcast(Eigen::array<int, 2>({{batch_size, 1}}));
  } else {
    g.device(place) = x;
  }
  const T* hidden_prev_data = hidden_prev.data<T>();
  const T* weight_data = weight.data<T>();
  T* gate_data = gate->data<T>();
  T* reset_hidden_prev_data = reset_hidden_prev->data<T>();
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  blas.GEMM(false,
            false,
            batch_size,
            2 * frame_size,
            frame_size,
            1,
            hidden_prev_data,
            frame_size,
            weight_data,
            frame_size * 2,
            1,
            gate_data,
            frame_size * 3);

  // calculate activated gate
  Eigen::array<int, 2> extents{{batch_size, frame_size}};
  Eigen::array<int, 2> u_offsets{{0, 0}};
  ACT_COMPUTE(gate_activation,
              place,
              g.slice(u_offsets, extents),
              g.slice(u_offsets, extents),
              dev_ctx.GetPlace());
  auto u = g.slice(u_offsets, extents);  // update gate
  Eigen::array<int, 2> r_offsets{{0, frame_size}};
  ACT_COMPUTE(gate_activation,
              place,
              g.slice(r_offsets, extents),
              g.slice(r_offsets, extents),
              dev_ctx.GetPlace());
  auto r = g.slice(r_offsets, extents);  // reset gate
  r_h_p.device(place) = r * h_p;         // reset previous hidden state
  blas.GEMM(false,
            false,
            batch_size,
            frame_size,
            frame_size,
            1,
            reset_hidden_prev_data,
            frame_size,
            weight_data + frame_size * frame_size * 2,
            frame_size,
            1,
            gate_data + frame_size * 2,
            frame_size * 3);

  Eigen::array<int, 2> c_offsets{{0, frame_size * 2}};
  ACT_COMPUTE(activation,
              place,
              g.slice(c_offsets, extents),
              g.slice(c_offsets, extents),
              dev_ctx.GetPlace());
  auto c = g.slice(c_offsets, extents);  // output candidate

  // calculate final output
  if (origin_mode) {
    h.device(place) = c + u * (h_p - c);  // (1 - u) * c + u * h_p
  } else {
    h.device(place) = u * (c - h_p) + h_p;  // u * c + (1 - u) * h_p
  }
}

template <typename T,
          typename Device,
          typename X,
          typename Y,
          typename DX,
          typename DY>
void ActGradCompute(
    const int act_type, const Device& d, X x, Y y, DX dx, DY dy) {
  // x is dummy and won't be used even in Relu(use y instead)
  if (act_type == identity)
    dx.device(d) = dy;
  else if (act_type == sigmoid)
    phi::funcs::SigmoidGradFunctor<T>()(d, x, y, dy, dx);
  else if (act_type == tanh)
    phi::funcs::TanhGradFunctor<T>()(d, x, y, dy, dx);
  else if (act_type == relu)
    phi::funcs::ReluGradFunctor<T>()(d, x, y, dy, dx);
  else
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported activation type, only supports identity, sigmoid, tanh "
        "and relu."));
}

#define ACT_GRAD_COMPUTE ActGradCompute<T>

template <typename T, typename Context>
void GRUUnitGradKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const DenseTensor& hidden_prev,
                       const DenseTensor& weight,
                       const paddle::optional<DenseTensor>& bias,
                       const DenseTensor& gate,
                       const DenseTensor& reset_hidden_prev,
                       const DenseTensor& hidden_grad,
                       int activation,
                       int gate_activation,
                       bool origin_mode,
                       DenseTensor* input_grad,
                       DenseTensor* hidden_prev_grad,
                       DenseTensor* weight_grad,
                       DenseTensor* bias_grad) {
  phi::DenseTensor gate_grad;
  phi::DenseTensor reset_hidden_prev_grad;

  const T* hidden_prev_data = hidden_prev.data<T>();
  const T* weight_data = weight.data<T>();
  gate_grad.Resize(input.dims());
  T* gate_grad_data = dev_ctx.template Alloc<T>(&gate_grad);
  const T* reset_hidden_prev_data = reset_hidden_prev.data<T>();
  reset_hidden_prev_grad.Resize(reset_hidden_prev.dims());
  T* reset_hidden_prev_grad_data =
      dev_ctx.template Alloc<T>(&reset_hidden_prev_grad);

  auto h_p = phi::EigenMatrix<T>::From(hidden_prev);
  auto g = phi::EigenMatrix<T>::From(gate);
  auto d_h = phi::EigenMatrix<T>::From(hidden_grad);
  auto d_g = phi::EigenMatrix<T>::From(gate_grad);
  auto d_r_h_p = phi::EigenMatrix<T>::From(reset_hidden_prev_grad);
  auto& place = *dev_ctx.eigen_device();

  int batch_size = input.dims()[0];
  int frame_size = hidden_prev.dims()[1];

  Eigen::array<int, 2> extents{{batch_size, frame_size}};
  Eigen::array<int, 2> u_offsets{{0, 0}};
  auto u = g.slice(u_offsets, extents);  // update gate
  Eigen::array<int, 2> r_offsets{{0, frame_size}};
  auto r = g.slice(r_offsets, extents);  // reset gate
  Eigen::array<int, 2> c_offsets{{0, frame_size * 2}};
  auto c = g.slice(c_offsets, extents);  // output candidate

  // backward for unactivated update gate
  if (origin_mode) {
    ACT_GRAD_COMPUTE(gate_activation,
                     place,
                     u,
                     u,
                     d_g.slice(u_offsets, extents),
                     d_h * (h_p - c));
    // backward for unactivated output candidate
    ACT_GRAD_COMPUTE(
        activation, place, c, c, d_g.slice(c_offsets, extents), d_h * (1 - u));
  } else {
    ACT_GRAD_COMPUTE(gate_activation,
                     place,
                     u,
                     u,
                     d_g.slice(u_offsets, extents),
                     d_h * (c - h_p));
    // backward for unactivated output candidate
    ACT_GRAD_COMPUTE(
        activation, place, c, c, d_g.slice(c_offsets, extents), d_h * u);
  }
  // backward for reset_hidden_prev
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  blas.GEMM(false,
            true,
            batch_size,
            frame_size,
            frame_size,
            1,
            gate_grad_data + frame_size * 2,
            frame_size * 3,
            weight_data + frame_size * frame_size * 2,
            frame_size,
            0,
            reset_hidden_prev_grad_data,
            frame_size);
  // backward for unactivated reset gate
  ACT_GRAD_COMPUTE(gate_activation,
                   place,
                   r,
                   r,
                   d_g.slice(r_offsets, extents),
                   d_r_h_p * h_p);
  // backward for weight
  if (weight_grad) {
    T* weight_grad_data = dev_ctx.template Alloc<T>(weight_grad);
    // backward for state_weight
    blas.GEMM(true,
              false,
              frame_size,
              frame_size,
              batch_size,
              1,
              reset_hidden_prev_data,
              frame_size,
              gate_grad_data + frame_size * 2,
              frame_size * 3,
              0,
              weight_grad_data + frame_size * frame_size * 2,
              frame_size);

    // backward for update_gate_weight and reset_gate_weight
    blas.GEMM(true,
              false,
              frame_size,
              frame_size * 2,
              batch_size,
              1,
              hidden_prev_data,
              frame_size,
              gate_grad_data,
              frame_size * 3,
              0,
              weight_grad_data,
              frame_size * 2);
  }
  // backward for hidden_prev
  if (hidden_prev_grad) {
    T* hidden_prev_grad_data = dev_ctx.template Alloc<T>(hidden_prev_grad);
    auto d_h_p = phi::EigenMatrix<T>::From(*hidden_prev_grad);
    if (origin_mode) {
      d_h_p.device(place) = d_r_h_p * r + d_h * u;
    } else {
      d_h_p.device(place) = d_r_h_p * r + d_h * (1 - u);
    }
    blas.GEMM(false,
              true,
              batch_size,
              frame_size,
              frame_size * 2,
              1,
              gate_grad_data,
              frame_size * 3,
              weight_data,
              frame_size * 2,
              1,
              hidden_prev_grad_data,
              frame_size);
  }
  // backward for input
  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    auto d_x = phi::EigenMatrix<T>::From(*input_grad);
    d_x.device(place) = d_g;
  }
  // backward for bias
  if (bias_grad) {
    dev_ctx.template Alloc<T>(bias_grad);
    auto d_b = phi::EigenVector<T>::Flatten(*bias_grad);
    d_b.device(place) = d_g.sum(Eigen::array<int, 1>({{0}}));
  }
}
}  // namespace phi
