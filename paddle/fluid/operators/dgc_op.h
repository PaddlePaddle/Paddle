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
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/platform/dynload/sparse_comm.h"

namespace paddle {
namespace operators {

inline float get_period_sparcity(const std::vector<float>& sparsity,
                                 float cur_step, float rampup_steps) {
  PADDLE_ENFORCE(static_cast<int>(cur_step) >= 0);

  size_t idx = static_cast<int>(cur_step * sparsity.size() / rampup_steps);
  if (idx >= sparsity.size()) {
    return 0.999;
  }

  PADDLE_ENFORCE(idx < sparsity.size());
  return sparsity[idx];
}

template <typename DeviceContext, typename T>
class DGCOpKernel : public framework::OpKernel<T> {
 public:
  // FIXME(gongwb): add gradient clipping.
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto u = ctx.Input<framework::Tensor>("U");
    auto v = ctx.Input<framework::Tensor>("V");
    auto g = ctx.Input<framework::Tensor>("Grad");

    // attrs
    float m = ctx.Attr<float>("m");
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");
    auto sparsity = ctx.Attr<std::vector<float>>("sparsity");
    auto rampup_begin_step = ctx.Attr<float>("rampup_begin_step");
    auto rampup_step = ctx.Attr<float>("rampup_step");

    // current step
    auto current_step_tensor = ctx.Input<framework::Tensor>("current_step");
    const float* current_step = current_step_tensor->data<float>();

    if (static_cast<int>(*current_step) < static_cast<int>(rampup_begin_step)) {
      VLOG(10) << "current_step:" << *current_step
               << " < rampup_begin_step:" << rampup_begin_step
               << " so does't use dgc";
      return;
    }

    float ratio =
        1 - get_period_sparcity(sparsity, static_cast<float>(*current_step),
                                rampup_step);
    PADDLE_ENFORCE(ratio > 0.0 && ratio < 1.0);
    int k = static_cast<int>(g->numel() * ratio);

    VLOG(10) << "m:" << m << ", use_nesterov:" << use_nesterov
             << ", rampup_begin_step:" << rampup_begin_step
             << ", rampup_step:" << rampup_step
             << ",  current_step:" << *current_step << ", ratio:" << ratio
             << ", k:" << k;

    auto k_out = ctx.Output<framework::Tensor>("k");
    T* k_out_data = k_out->data<T>();
    *k_out_data = k;

    auto u_out = ctx.Output<framework::Tensor>("U_out");
    auto v_out = ctx.Output<framework::Tensor>("V_out");
    auto encode_grad_out = ctx.Output<framework::Tensor>("EncodeGrad");

    // FIXME(gongwb): use cublas.
    auto u_out_e = framework::EigenVector<T>::Flatten(*u_out);
    auto u_e = framework::EigenVector<T>::Flatten(*u);
    auto g_e = framework::EigenVector<T>::Flatten(*g);
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto& eigen_ctx = *dev_ctx.eigen_device();
    if (use_nesterov) {
      // u = m * (u + g)
      u_out_e.device(eigen_ctx) = m * (u_e + g_e);

      // v = u + v + g
      ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(
          ctx, u, v, 0, AddFunctor<T>(), v_out);

      ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(
          ctx, g, v, 0, AddFunctor<T>(), v_out);
    } else {
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

    int buf_size = paddle::platform::dynload::get_dgc_buffer_size(k);
    auto& allocator = platform::DeviceTemporaryAllocator::Instance().Get(
        ctx.GetPlace(), dev_ctx.stream());
    auto tmp_ious_data = allocator.Allocate(buf_size);
    void* buf = reinterpret_cast<void*>(tmp_ious_data->ptr());

    if (!paddle::platform::dynload::dgc_k_select(
            static_cast<void*>(encode_grad_out_data), k, v_out_data,
            static_cast<int>(v_out->numel()), buf, dev_ctx.stream(),
            u_out_data)) {
      LOG(FATAL) << "v_out numel:" << v_out->numel();
    }

    auto grad_out = ctx.Output<framework::Tensor>("Grad_out");
    math::SetConstant<DeviceContext, T> tset;
    tset(dev_ctx, grad_out, static_cast<T>(0));
  }
};
}  // namespace operators
}  // namespace paddle
