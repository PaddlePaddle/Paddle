/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
class LogSoftmaxMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T, dnnl::logsoftmax_forward> {
 public:
  LogSoftmaxMKLDNNHandler(const dnnl::engine mkldnn_engine,
                          platform::Place cpu_place, const Tensor* x,
                          const int axis)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::logsoftmax_forward>(
            mkldnn_engine, cpu_place) {
    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_inference,
                                            x->mem_desc(), axis);
  }
};

template <typename T>
class LogSoftmaxMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const Tensor* x = ctx.Input<Tensor>("X");
    Tensor* out = ctx.Output<Tensor>("Out");

    int axis = ctx.Attr<int>("axis");
    axis = axis >= 0 ? axis : x->dims().size() + axis;

    LogSoftmaxMKLDNNHandler<T> handler(mkldnn_engine, ctx.GetPlace(), x, axis);

    auto src_memory_p = handler.AcquireSrcMemory(x);
    auto dst_memory_p = handler.AcquireDstMemory(out);

    auto logsoftmax_p = handler.AcquireForwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    logsoftmax_p->execute(astream, {{DNNL_ARG_SRC, *src_memory_p},
                                    {DNNL_ARG_DST, *dst_memory_p}});
    astream.wait();

    out->set_mem_desc(dst_memory_p->get_desc());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(log_softmax, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::LogSoftmaxMKLDNNKernel<float>,
                   ops::LogSoftmaxMKLDNNKernel<paddle::platform::bfloat16>);
