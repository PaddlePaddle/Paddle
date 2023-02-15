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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {

enum GRUActivationType { identity = 0, sigmoid = 1, tanh = 2, relu = 3 };

template <typename DeviceContext, typename T>
class GRUUnitKernel : public framework::OpKernel<T> {
 public:
  template <typename Device, typename X, typename Y>
  void ActCompute(const int act_type,
                  const Device& d,
                  X x,
                  Y y,
                  platform::Place place) const {
    if (act_type == identity) {
      y.device(d) = x;
    } else if (act_type == sigmoid) {
      SigmoidFunctor<T>()(d, x, y);
    } else if (act_type == tanh) {
      TanhFunctor<T>()(d, x, y);
    } else if (act_type == relu) {
      if (place == platform::CPUPlace())
        ReluCPUFunctor<T>()(d, x, y);
      else
        ReluCUDAFunctor<T>()(d, x, y);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported activation type, only supports identity, sigmoid, tanh "
          "and relu."));
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<phi::DenseTensor>("Input");
    auto* hidden_prev = context.Input<phi::DenseTensor>("HiddenPrev");
    auto* weight = context.Input<phi::DenseTensor>("Weight");
    auto* bias = context.Input<phi::DenseTensor>("Bias");
    auto* gate = context.Output<phi::DenseTensor>("Gate");
    gate->mutable_data<T>(context.GetPlace());
    auto* reset_hidden_prev =
        context.Output<phi::DenseTensor>("ResetHiddenPrev");
    reset_hidden_prev->mutable_data<T>(context.GetPlace());
    auto* hidden = context.Output<phi::DenseTensor>("Hidden");
    hidden->mutable_data<T>(context.GetPlace());

    int batch_size = input->dims()[0];
    int frame_size = hidden_prev->dims()[1];

    auto x = framework::EigenMatrix<T>::From(*input);
    auto h_p = framework::EigenMatrix<T>::From(*hidden_prev);
    auto g = framework::EigenMatrix<T>::From(*gate);
    auto r_h_p = framework::EigenMatrix<T>::From(*reset_hidden_prev);
    auto h = framework::EigenMatrix<T>::From(*hidden);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    // calculate unactivated gate outputs
    if (bias) {
      auto b = framework::EigenMatrix<T>::From(*bias);
      g.device(place) =
          x + b.reshape(Eigen::array<int, 2>({{1, frame_size * 3}}))
                  .broadcast(Eigen::array<int, 2>({{batch_size, 1}}));
    } else {
      g.device(place) = x;
    }
    const T* hidden_prev_data = hidden_prev->data<T>();
    const T* weight_data = weight->data<T>();
    T* gate_data = gate->data<T>();
    T* reset_hidden_prev_data = reset_hidden_prev->data<T>();
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(context);
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

    // calculate activited gate
    Eigen::array<int, 2> extents{{batch_size, frame_size}};
    Eigen::array<int, 2> u_offsets{{0, 0}};
    ActCompute(context.Attr<int>("gate_activation"),
               place,
               g.slice(u_offsets, extents),
               g.slice(u_offsets, extents),
               context.GetPlace());
    auto u = g.slice(u_offsets, extents);  // update gate
    Eigen::array<int, 2> r_offsets{{0, frame_size}};
    ActCompute(context.Attr<int>("gate_activation"),
               place,
               g.slice(r_offsets, extents),
               g.slice(r_offsets, extents),
               context.GetPlace());
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
    ActCompute(context.Attr<int>("activation"),
               place,
               g.slice(c_offsets, extents),
               g.slice(c_offsets, extents),
               context.GetPlace());
    auto c = g.slice(c_offsets, extents);  // output candidate

    // calculate final output
    if (context.Attr<bool>("origin_mode")) {
      h.device(place) = c + u * (h_p - c);  // (1 - u) * c + u * h_p
    } else {
      h.device(place) = u * (c - h_p) + h_p;  // u * c + (1 - u) * h_p
    }
  }
};

template <typename DeviceContext, typename T>
class GRUUnitGradKernel : public framework::OpKernel<T> {
 public:
  template <typename Device, typename X, typename Y, typename DX, typename DY>
  void ActGradCompute(
      const int act_type, const Device& d, X x, Y y, DX dx, DY dy) const {
    // x is dummy and won't be used even in Relu(use y instead)
    if (act_type == identity)
      dx.device(d) = dy;
    else if (act_type == sigmoid)
      SigmoidGradFunctor<T>()(d, x, y, dy, dx);
    else if (act_type == tanh)
      TanhGradFunctor<T>()(d, x, y, dy, dx);
    else if (act_type == relu)
      ReluGradFunctor<T>()(d, x, y, dy, dx);
    else
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported activation type, only supports identity, sigmoid, tanh "
          "and relu."));
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<phi::DenseTensor>("Input");
    auto* hidden_prev = context.Input<phi::DenseTensor>("HiddenPrev");
    auto* weight = context.Input<phi::DenseTensor>("Weight");
    auto* gate = context.Input<phi::DenseTensor>("Gate");
    auto* reset_hidden_prev =
        context.Input<phi::DenseTensor>("ResetHiddenPrev");
    auto* hidden_grad =
        context.Input<phi::DenseTensor>(framework::GradVarName("Hidden"));
    auto* input_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("Input"));
    auto* hidden_prev_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("HiddenPrev"));
    auto* weight_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("Weight"));
    auto* bias_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("Bias"));
    phi::DenseTensor gate_grad;
    phi::DenseTensor reset_hidden_prev_grad;

    const T* hidden_prev_data = hidden_prev->data<T>();
    const T* weight_data = weight->data<T>();
    T* gate_grad_data =
        gate_grad.mutable_data<T>(input->dims(), context.GetPlace());
    const T* reset_hidden_prev_data = reset_hidden_prev->data<T>();
    T* reset_hidden_prev_grad_data = reset_hidden_prev_grad.mutable_data<T>(
        reset_hidden_prev->dims(), context.GetPlace());

    auto h_p = framework::EigenMatrix<T>::From(*hidden_prev);
    auto g = framework::EigenMatrix<T>::From(*gate);
    auto d_h = framework::EigenMatrix<T>::From(*hidden_grad);
    auto d_g = framework::EigenMatrix<T>::From(gate_grad);
    auto d_r_h_p = framework::EigenMatrix<T>::From(reset_hidden_prev_grad);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    int batch_size = input->dims()[0];
    int frame_size = hidden_prev->dims()[1];

    Eigen::array<int, 2> extents{{batch_size, frame_size}};
    Eigen::array<int, 2> u_offsets{{0, 0}};
    auto u = g.slice(u_offsets, extents);  // update gate
    Eigen::array<int, 2> r_offsets{{0, frame_size}};
    auto r = g.slice(r_offsets, extents);  // reset gate
    Eigen::array<int, 2> c_offsets{{0, frame_size * 2}};
    auto c = g.slice(c_offsets, extents);  // output candidate

    // backward for unactivated update gate
    if (context.Attr<bool>("origin_mode")) {
      ActGradCompute(context.Attr<int>("gate_activation"),
                     place,
                     u,
                     u,
                     d_g.slice(u_offsets, extents),
                     d_h * (h_p - c));
      // backward for unactivated output candidate
      ActGradCompute(context.Attr<int>("activation"),
                     place,
                     c,
                     c,
                     d_g.slice(c_offsets, extents),
                     d_h * (1 - u));
    } else {
      ActGradCompute(context.Attr<int>("gate_activation"),
                     place,
                     u,
                     u,
                     d_g.slice(u_offsets, extents),
                     d_h * (c - h_p));
      // backward for unactivated output candidate
      ActGradCompute(context.Attr<int>("activation"),
                     place,
                     c,
                     c,
                     d_g.slice(c_offsets, extents),
                     d_h * u);
    }
    // backward for reset_hidden_prev
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(context);
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
    ActGradCompute(context.Attr<int>("gate_activation"),
                   place,
                   r,
                   r,
                   d_g.slice(r_offsets, extents),
                   d_r_h_p * h_p);
    // backward for weight
    if (weight_grad) {
      T* weight_grad_data = weight_grad->mutable_data<T>(context.GetPlace());
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
      T* hidden_prev_grad_data =
          hidden_prev_grad->mutable_data<T>(context.GetPlace());
      auto d_h_p = framework::EigenMatrix<T>::From(*hidden_prev_grad);
      if (context.Attr<bool>("origin_mode")) {
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
      input_grad->mutable_data<T>(context.GetPlace());
      auto d_x = framework::EigenMatrix<T>::From(*input_grad);
      d_x.device(place) = d_g;
    }
    // backward for bias
    if (bias_grad) {
      bias_grad->mutable_data<T>(context.GetPlace());
      auto d_b = framework::EigenVector<T>::Flatten(*bias_grad);
      d_b.device(place) = d_g.sum(Eigen::array<int, 1>({{0}}));
    }
  }
};

}  // namespace operators
}  // namespace paddle
