/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/operators/math/math_function.h"

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class GRUUnitKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("input");
    auto* hidden_prev = context.Input<Tensor>("hidden_prev");
    auto* weight = context.Input<Tensor>("weight");
    auto* bias = context.Input<Tensor>("bias");
    auto* gate = context.Output<Tensor>("gate");
    gate->mutable_data<T>(context.GetPlace());
    auto* reset_hidden_prev = context.Output<Tensor>("reset_hidden_prev");
    reset_hidden_prev->mutable_data<T>(context.GetPlace());
    auto* hidden = context.Output<Tensor>("hidden");
    hidden->mutable_data<T>(context.GetPlace());

    int batch_size = input->dims()[0];
    int frame_size = hidden_prev->dims()[1];

    auto x = EigenMatrix<T>::From(*input);
    auto h_p = EigenMatrix<T>::From(*hidden_prev);
    auto b = EigenMatrix<T>::From(*bias);
    auto g = EigenMatrix<T>::From(*gate);
    auto r_h_p = EigenMatrix<T>::From(*reset_hidden_prev);
    auto h = EigenMatrix<T>::From(*hidden);
    auto place = context.GetEigenDevice<Place>();

    // calculate unactivated gate outputs
    g.device(place) = x +
                      b.reshape(Eigen::array<int, 2>({{1, frame_size * 3}}))
                          .broadcast(Eigen::array<int, 2>({{batch_size, 1}}));
    const T* hidden_prev_data = hidden_prev->data<T>();
    const T* weight_data = weight->data<T>();
    T* gate_data = gate->data<T>();
    T* reset_hidden_prev_data = reset_hidden_prev->data<T>();
    math::gemm<Place, T>(context.device_context(), false, false, batch_size,
                         2 * frame_size, frame_size, 1, hidden_prev_data,
                         frame_size, weight_data, frame_size * 2, 1, gate_data,
                         frame_size * 3);

    // calculate activited gate
    Eigen::array<int, 2> extents({{batch_size, frame_size}});
    Eigen::array<int, 2> u_offsets({{0, 0}});
    g.slice(u_offsets, extents).device(place) =
        g.slice(u_offsets, extents).sigmoid();
    auto u = g.slice(u_offsets, extents);  // update gate
    Eigen::array<int, 2> r_offsets({{0, frame_size}});
    g.slice(r_offsets, extents).device(place) =
        g.slice(r_offsets, extents).sigmoid();
    auto r = g.slice(r_offsets, extents);  // reset gate
    r_h_p.device(place) = r * h_p;         // reset previous hidden state
    math::gemm<Place, T>(context.device_context(), false, false, batch_size,
                         frame_size, frame_size, 1, reset_hidden_prev_data,
                         frame_size, weight_data + frame_size * frame_size * 2,
                         frame_size, 1, gate_data + frame_size * 2,
                         frame_size * 3);

    Eigen::array<int, 2> c_offsets({{0, frame_size * 2}});
    g.slice(c_offsets, extents).device(place) =
        g.slice(c_offsets, extents).tanh();
    auto c = g.slice(c_offsets, extents);  // output candidate

    // calculate final output
    h.device(place) = u * (h_p - c) + c;
  }
};

template <typename Place, typename T>
class GRUUnitGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("input");
    auto* hidden_prev = context.Input<Tensor>("hidden_prev");
    auto* weight = context.Input<Tensor>("weight");
    auto* gate = context.Input<Tensor>("gate");
    auto* reset_hidden_prev = context.Input<Tensor>("reset_hidden_prev");
    auto* hidden_grad = context.Input<Tensor>(framework::GradVarName("hidden"));
    auto* input_grad = context.Output<Tensor>(framework::GradVarName("input"));
    auto* hidden_prev_grad =
        context.Output<Tensor>(framework::GradVarName("hidden_prev"));
    auto* weight_grad =
        context.Output<Tensor>(framework::GradVarName("weight"));
    auto* bias_grad = context.Output<Tensor>(framework::GradVarName("bias"));
    input_grad->mutable_data<T>(context.GetPlace());
    hidden_prev_grad->mutable_data<T>(context.GetPlace());
    weight_grad->mutable_data<T>(context.GetPlace());
    bias_grad->mutable_data<T>(context.GetPlace());
    Tensor gate_grad;
    gate_grad.mutable_data<T>(input->dims(), context.GetPlace());
    Tensor reset_hidden_prev_grad;
    reset_hidden_prev_grad.mutable_data<T>(reset_hidden_prev->dims(),
                                           context.GetPlace());

    int batch_size = input->dims()[0];
    int frame_size = hidden_prev->dims()[1];

    const T* hidden_prev_data = hidden_prev->data<T>();
    T* hidden_prev_grad_data = hidden_prev_grad->data<T>();
    const T* weight_data = weight->data<T>();
    T* weight_grad_data = weight_grad->data<T>();
    T* gate_grad_data = gate_grad.data<T>();
    const T* reset_hidden_prev_data = reset_hidden_prev->data<T>();
    T* reset_hidden_prev_grad_data = reset_hidden_prev_grad.data<T>();

    auto h_p = EigenMatrix<T>::From(*hidden_prev);
    auto g = EigenMatrix<T>::From(*gate);
    auto d_h = EigenMatrix<T>::From(*hidden_grad);
    auto d_x = EigenMatrix<T>::From(*input_grad);
    auto d_h_p = EigenMatrix<T>::From(*hidden_prev_grad);
    auto d_b = EigenMatrix<T>::From(*bias_grad);
    auto d_g = EigenMatrix<T>::From(gate_grad);
    auto d_r_h_p = EigenMatrix<T>::From(reset_hidden_prev_grad);
    auto place = context.GetEigenDevice<Place>();

    Eigen::array<int, 2> extents({{batch_size, frame_size}});
    Eigen::array<int, 2> u_offsets({{0, 0}});
    auto u = g.slice(u_offsets, extents);  // update gate
    Eigen::array<int, 2> r_offsets({{0, frame_size}});
    auto r = g.slice(r_offsets, extents);  // reset gate
    Eigen::array<int, 2> c_offsets({{0, frame_size * 2}});
    auto c = g.slice(c_offsets, extents);  // output candidate

    // backward for unactivated update gate
    d_g.slice(u_offsets, extents).device(place) =
        d_h * (h_p - c) * u * (u.constant(T(1)) - u);
    // backward for unactivated output candidate
    d_g.slice(c_offsets, extents).device(place) =
        d_h * (u.constant(T(1)) - u) * (c.constant(T(1)) - c * c);
    // backward for reset_hidden_prev
    math::gemm<Place, T>(context.device_context(), false, true, batch_size,
                         frame_size, frame_size, 1,
                         gate_grad_data + frame_size * 2, frame_size * 3,
                         weight_data + frame_size * frame_size * 2, frame_size,
                         0, reset_hidden_prev_grad_data, frame_size);
    // backward for state_weight
    math::gemm<Place, T>(
        context.device_context(), true, false, frame_size, frame_size,
        batch_size, 1, reset_hidden_prev_data, frame_size,
        gate_grad_data + frame_size * 2, frame_size * 3, 0,
        weight_grad_data + frame_size * frame_size * 2, frame_size);
    // backward for unactivated reset gate
    d_g.slice(r_offsets, extents).device(place) =
        d_r_h_p * h_p * r * (r.constant(T(1)) - r);
    // backward for update_gate_weight and reset_gate_weight
    math::gemm<Place, T>(context.device_context(), true, false, frame_size,
                         frame_size * 2, batch_size, 1, hidden_prev_data,
                         frame_size, gate_grad_data, frame_size * 3, 0,
                         weight_grad_data, frame_size * 2);
    // backward for hidden_prev
    d_h_p.device(place) = d_r_h_p * r + d_h * u;
    math::gemm<Place, T>(context.device_context(), false, true, batch_size,
                         frame_size, frame_size * 2, 1, gate_grad_data,
                         frame_size * 3, weight_data, frame_size * 2, 1,
                         hidden_prev_grad_data, frame_size);
    // backward for input
    d_x.device(place) = d_g;
    // backward for bias
    d_b.device(place) = d_g.sum(Eigen::array<int, 1>({{0}}));
  }
};

}  // namespace operators
}  // namespace paddle
