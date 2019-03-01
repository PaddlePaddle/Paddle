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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/detail/activation_functions.h"
#include "paddle/fluid/operators/math/fusion_bidirectional_gru_compute.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class FusionBidirectionalGRUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<LoDTensor>("X");
    const T *x_data = x->data<T>();
    auto *out = ctx.Output<Tensor>("Out");
    T *out_data = out->mutable_data<T>(ctx.GetPlace());

    const Tensor *init_h0 = ctx.Input<Tensor>("InitH0");
    const Tensor *init_h1 = ctx.Input<Tensor>("InitH1");

    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    math::SetConstant<paddle::platform::CUDADeviceContext, T> zero;

    T *h0p = const_cast<T *>(init_h0->data<T>());
    T *h1p = const_cast<T *>(init_h1->data<T>());

    auto w0x = ctx.Input<Tensor>("Weight0X");
    const T *w0x_data = w0x->data<T>();
    auto w1x = ctx.Input<Tensor>("Weight1X");
    const T *w1x_data = w1x->data<T>();
    auto w2x = ctx.Input<Tensor>("Weight2X");
    const T *w2x_data = w2x->data<T>();
    auto w3x = ctx.Input<Tensor>("Weight3X");
    const T *w3x_data = w3x->data<T>();

    auto w0h = ctx.Input<Tensor>("Weight0H");
    const T *w0h_data = w0h->data<T>();
    auto w1h = ctx.Input<Tensor>("Weight1H");
    const T *w1h_data = w1h->data<T>();

    const T *bias0x_data = nullptr;
    const T *bias1x_data = nullptr;

    auto *bias0x = ctx.Input<LoDTensor>("Bias0X");
    if (bias0x != nullptr) bias0x_data = bias0x->data<T>();
    auto *bias1x = ctx.Input<LoDTensor>("Bias1X");
    if (bias1x != nullptr) bias1x_data = bias1x->data<T>();

    auto *bias0h = ctx.Input<LoDTensor>("Bias0H");
    const T *bias0h_data = bias0h->data<T>();
    auto *bias1h = ctx.Input<LoDTensor>("Bias1H");
    const T *bias1h_data = bias1h->data<T>();

    auto *mul_out0 = ctx.Output<Tensor>("mul_out0");
    T *m_out0 = mul_out0->mutable_data<T>(ctx.GetPlace());

    auto *mul_out1 = ctx.Output<Tensor>("mul_out1");
    T *m_out1 = mul_out1->mutable_data<T>(ctx.GetPlace());

    auto *gru_out0 = ctx.Output<Tensor>("gru_out0");
    T *g_out0 = gru_out0->mutable_data<T>(ctx.GetPlace());

    auto *gru_out1 = ctx.Output<Tensor>("gru_out1");
    T *g_out1 = gru_out1->mutable_data<T>(ctx.GetPlace());
    zero(dev_ctx, gru_out1, static_cast<T>(0.0));

    framework::DDim x_dim = x->dims();
    framework::DDim wx_dim = w0x->dims();

    framework::DDim w2x_dim = w2x->dims();

    auto active_gate = math::detail::GetActivationType(
        ctx.Attr<std::string>("gate_activation"));

    auto active_node =
        math::detail::GetActivationType(ctx.Attr<std::string>("activation"));

    auto *gate0_tensor = ctx.Output<Tensor>("gate0");
    auto *gate0 = gate0_tensor->mutable_data<T>(ctx.GetPlace());
    zero(dev_ctx, gate0_tensor, static_cast<T>(0.0));
    auto *gate1_tensor = ctx.Output<Tensor>("gate1");
    auto *gate1 = gate1_tensor->mutable_data<T>(ctx.GetPlace());
    zero(dev_ctx, gate1_tensor, static_cast<T>(0.0));

    int reverse = ctx.Attr<int>("reverse");

    math::FusionGRUMetaValue<T> value;
    value.x = x_data;
    value.wx0 = w0x_data;
    value.wx1 = w1x_data;
    value.wx2 = w2x_data;
    value.wx3 = w3x_data;
    value.bias_x0 = bias0x_data;
    value.bias_x1 = bias1x_data;
    value.mul_o0 = m_out0;
    value.mul_o1 = m_out1;
    value.bias_h0 = bias0h_data;
    value.bias_h1 = bias1h_data;

    value.wh0 = w0h_data;
    value.wh1 = w1h_data;
    value.hp0 = h0p;
    value.hp1 = h1p;
    value.gate0 = gate0;
    value.gate1 = gate1;
    value.gru_o0 = g_out0;
    value.gru_o1 = g_out1;
    value.out = out_data;

    math::FusionBidirectionalGRUFunctor<DeviceContext, T>::compute(
        dev_ctx, value, wx_dim[1], x_dim[0], wx_dim[0], w2x_dim[1], active_gate,
        active_node, reverse);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fusion_bidirectional_gru,
    ops::FusionBidirectionalGRUKernel<paddle::platform::CUDADeviceContext,
                                      float>,
    ops::FusionBidirectionalGRUKernel<paddle::platform::CUDADeviceContext,
                                      double>);
