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
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/k_select/k_select.h"

namespace paddle {
namespace operators {

inline float get_period_sparcity(float cur_step, float rampup_steps) {
  if (static_cast<int>(cur_step) <= 0) {
    return 0.75;
  }

  std::vector<float> a({0.75, 0.9375, 0.984375, 0.996, 0.999});
  float rate = cur_step / rampup_steps;
  int idx = static_cast<int>(rate / 0.2);
  if (idx >= a.size()) {
    return 0.999;
  }

  PADDLE_ENFORCE(idx >= 0 && idx < a.size());
  return a[idx];
}

template <typename DeviceContext, typename T>
class DGCOpKernel : public framework::OpKernel<T> {
 public:
  // FIXME(gongwb): add gradient clipping.
  void Compute(const framework::ExecutionContext& ctx) const override {
    VLOG(10) << "begin run dgc op kernel";
    auto u = ctx.Input<framework::Tensor>("U");
    auto v = ctx.Input<framework::Tensor>("V");
    auto g = ctx.Input<framework::Tensor>("Grad");
    float m = ctx.Attr<float>("m");
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");

    auto current_step_tensor = ctx.Input<framework::Tensor>("current_step");
    auto rampup_step_tensor = ctx.Input<framework::Tensor>("rampup_step");
    const float* current_step = current_step_tensor->data<float>();
    const float* rampup_step = rampup_step_tensor->data<float>();
    VLOG(10) << "current_step_tensor:" << current_step_tensor
             << ", rampup_step_tensor:" << rampup_step_tensor
             << ", current_step:" << current_step
             << ", rampup_step:" << rampup_step;
    float ratio = 1 - get_period_sparcity(static_cast<float>(*current_step),
                                          static_cast<float>(*rampup_step));
    int k = static_cast<int>(g->numel() * ratio);
    VLOG(10) << "rampup_step:" << *rampup_step
             << ", current_step:" << *current_step << ", ratio:" << ratio
             << ", k:" << k;

    auto u_out = ctx.Output<framework::Tensor>("U_out");
    auto v_out = ctx.Output<framework::Tensor>("V_out");
    auto encode_grad_out = ctx.Output<framework::Tensor>("EncodeGrad");

    // FIXME(gognwb): use cublas.
    auto u_out_e = framework::EigenVector<T>::Flatten(*u_out);
    auto u_e = framework::EigenVector<T>::Flatten(*u);
    auto g_e = framework::EigenVector<T>::Flatten(*g);
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto& eigen_ctx = *dev_ctx.eigen_device();
    if(use_nesterov){
        // u = m * (u + g)
        u_out_e.device(eigen_ctx) = m * (u_e + g_e);

        // v = u + v + g
        ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(
                ctx, u, v, 0, AddFunctor<T>(), v_out);

        ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(
                ctx, g, v, 0, AddFunctor<T>(), v_out);
    }else{
        // u = m * u + g
        u_out_e.device(eigen_ctx) = m * u_e + g_e;

        // v = u + v
        ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(
                ctx, u, v, 0, AddFunctor<T>(), v_out);
    }

    T* v_out_data = v_out->mutable_data<T>(ctx.GetPlace());
    T* u_out_data = u_out->mutable_data<T>(ctx.GetPlace());
    T* encode_grad_out_data = encode_grad_out->mutable_data<T>(
        framework::DDim{2 * k}, ctx.GetPlace());

    auto buf_out = ctx.Output<framework::Tensor>("Encoded_buf");
    void* buf = static_cast<void*>(buf_out->mutable_data<T>(ctx.GetPlace()));
    PADDLE_ENFORCE(k_select(v_out_data, static_cast<int>(v_out->numel()),
                            static_cast<void*>(encode_grad_out_data), buf, k, 0,
                            dev_ctx.stream(), u_out_data));

    // PADDLE_ENFORCE(k_select(v_out_data, static_cast<int>(v_out->numel()),
    // static_cast<void*>(encode_grad_out_data), k, dev_ctx.stream()));

    /*
    int buf_size = get_buffer_size(k) * sizeof(T);
    auto& allocator = platform::DeviceTemporaryAllocator::Instance().Get(
        ctx.GetPlace(), dev_ctx.stream());
    auto tmp_ious_data = allocator.Allocate(buf_size);
    void* buf = reinterpret_cast<void*>(tmp_ious_data->ptr());

    k_select(v_out_data,
        static_cast<int>(v_out->numel()),
        static_cast<void*>(encode_grad_out_data),
        buf, k, 0, dev_ctx.stream(), u_out_data);
    */
  }
};
}  // namespace operators
}  // namespace paddle
