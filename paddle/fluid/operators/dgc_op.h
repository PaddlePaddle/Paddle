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

template <typename DeviceContext, typename T>
class DGCOpKernel : public framework::OpKernel<T> {
 public:
  // FIXME(gongwb): add gradient clipping.
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto u = ctx.Input<framework::Tensor>("U");
    auto v = ctx.Input<framework::Tensor>("V");
    auto g = ctx.Input<framework::Tensor>("Grad");
    auto local_g = ctx.Input<framework::Tensor>("GradLocal");
    float m = static_cast<T>(ctx.Attr<float>("m"));
    float ratio = static_cast<T>(ctx.Attr<float>("ratio"));
    int k = static_cast<int>(g.numel() * ratio);

    auto u_out = ctx.Output<framework::Tensor>("U");
    auto v_out = ctx.Output<framework::Tensor>("V");
    // auto g_out = ctx.Output<framework::Tensor>("Grad");
    auto local_g_out = ctx.Input<framework::Tensor>("GradLocal");
    auto g_out = ctx.Output<framework::Tensor>("EncodeGradient");

    // local_g = local_g + g
    elementwise_add<DeviceContext, T>(ctx, local_g, g, local_g_out);

    // FIXME(gognwb): use cublas.
    // u = m * u + g
    auto u_out_e = framework::EigenVector<T>::Flatten(*u_out);
    auto u_e = framework::EigenVector<T>::Flatten(*u);
    auto g_e = framework::EigenVector<T>::Flatten(*g);
    auto& dev_ctx =
        *ctx.template device_context<DeviceContext>().eigen_device();
    u_out_e.device(dev_ctx) = m * u_e + g_e;

    // v = u + v
    elementwise_add<DeviceContext, T>(ctx, u, v, v_out);

    // getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks,
    // int &threads);
    int buffbytes = 2 * MAX_BLOCKS * MAX_THREADS * sizeof(int);

    // Temporary memory
    auto& allocator =
        platform::DeviceTemporaryAllocator::Instance().Get(dev_ctx);
    auto tmp_ious_data = allocator.Allocate(buffbytes);
    void* buf = tmp_ious_data->ptr();

    k_select(v_out.mutable_data<T>(), v_out.numel(), g_out.mutable_data<T>(),
             buf, k, dev_ctx.stream(), u_out.mutable_data<T>());
  }
};
}  // namespace operators
}  // namespace paddle
