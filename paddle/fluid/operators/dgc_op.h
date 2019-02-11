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
    VLOG(10) << "begin run dgc op kernel";
    auto u = ctx.Input<framework::Tensor>("U");
    auto v = ctx.Input<framework::Tensor>("V");
    auto g = ctx.Input<framework::Tensor>("Grad");
    float m = ctx.Attr<float>("m");
    float ratio = ctx.Attr<float>("ratio");
    int k = static_cast<int>(g->numel() * ratio);

    auto u_out = ctx.Output<framework::Tensor>("U_out");
    auto v_out = ctx.Output<framework::Tensor>("V_out");
    auto encode_grad_out = ctx.Output<framework::Tensor>("EncodeGrad");

    // FIXME(gognwb): use cublas.
    // u = m * u + g
    auto u_out_e = framework::EigenVector<T>::Flatten(*u_out);
    auto u_e = framework::EigenVector<T>::Flatten(*u);
    auto g_e = framework::EigenVector<T>::Flatten(*g);
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto& eigen_ctx = *dev_ctx.eigen_device();
    u_out_e.device(eigen_ctx) = m * u_e + g_e;

    // v = u + v
    ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(
        ctx, u, v, 0, AddFunctor<T>(), v_out);

    T* v_out_data = v_out->mutable_data<T>(ctx.GetPlace());
    T* u_out_data = u_out->mutable_data<T>(ctx.GetPlace());
    T* encode_grad_out_data = encode_grad_out->mutable_data<T>(ctx.GetPlace());

    auto buf_out = ctx.Output<framework::Tensor>("Encoded_buf");
    void* buf = static_cast<void*>(buf_out->mutable_data<T>(ctx.GetPlace()));
    k_select(v_out_data, static_cast<int>(v_out->numel()),
             static_cast<void*>(encode_grad_out_data), k, dev_ctx.stream());
    /*
    k_select(v_out_data,
        static_cast<int>(v_out->numel()),
        static_cast<void*>(encode_grad_out_data),
        buf, k, 0, dev_ctx.stream(), u_out_data);
        */
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

    /*
    int dev_id = 0;
    auto& allocator = platform::DeviceTemporaryAllocator::Instance().Get(
        ctx.GetPlace(), dev_ctx.stream());
    auto tmp_ious_data = allocator.Allocate(buf_size);

    bool malloc = false;
    bool wait = false;

    if (!malloc) {
      buf = reinterpret_cast<void*>(tmp_ious_data->ptr());
    } else {
      int dev_id = boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()).device;
      PADDLE_ENFORCE(cudaSetDevice(dev_id));
      buf = nullptr;
      cudaMalloc(&buf, buf_size);
      PADDLE_ENFORCE(buf);
    }

    VLOG(10) << "k:" << k << "encode_grad_out dims:" << encode_grad_out->dims()
             << ", buf_size:" << buf_size << ", v_out dims:" << v_out->dims()
             << ", u_out dims:" << u_out->dims() << ", devid:" << dev_id;


    k_select(v_out_data, static_cast<int>(v_out->numel()),
             static_cast<void*>(encode_grad_out_data), k, dev_ctx.stream());

    cudaMemsetAsync(buf, -1, buf_size, dev_ctx.stream());
    if (wait) dev_ctx.Wait();

    if (malloc) {
      cudaFree(buf);
    }
    */

    /*
    operators::math::SetConstant<DeviceContext, T> constant_functor;
    constant_functor(dev_ctx, encode_grad_out, static_cast<T>(0));
    constant_functor(dev_ctx, u_out, static_cast<T>(0));
    constant_functor(dev_ctx, v_out, static_cast<T>(0));
    */
    // Temporary memory
    /*
    auto& allocator = platform::DeviceTemporaryAllocator::Instance().Get(
        ctx.GetPlace(), dev_ctx.stream());
    auto tmp_ious_data = allocator.Allocate(buf_size);
    void* buf = reinterpret_cast<void*>(tmp_ious_data->ptr());
     */
    /*
    PADDLE_ENFORCE(cudaSetDevice(dev_id));
    void* buf = nullptr;
    cudaMalloc(&buf, buf_size);
    PADDLE_ENFORCE(buf);
    */
    /*
    framework::Tensor tmp;
    int buf_count = get_buffer_size(k)  * 10;
    void* buf = reinterpret_cast<void*>
    (tmp.Resize(framework::DDim({buf_count})).mutable_data<T>(ctx.GetPlace()));
    std::cout << buf << std::endl;
    */

    /*
    k_select(
        v_out->mutable_data<T>(ctx.GetPlace()),
        static_cast<int>(v_out->numel()),
        static_cast<void*>(encode_grad_out->mutable_data<T>(ctx.GetPlace())),
        buf, k, 0, dev_ctx.stream(), u_out->mutable_data<T>(ctx.GetPlace()));
        */
    // constant_functor(dev_ctx, &tmp, static_cast<T>(0));
    /*
    cudaMemsetAsync(buf, 0, tmp.numel(), dev_ctx.stream());
    dev_ctx.Wait();
    */

    // cudaFree(buf);
  }
};
}  // namespace operators
}  // namespace paddle
