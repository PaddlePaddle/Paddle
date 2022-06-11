/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

using paddle::framework::Tensor;

template <typename T>
class ScaleMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    bool is_inplaced = x->IsSharedBufferWith(*out);

    platform::ActivationMKLDNNHandler<T> handler(
        dnnl::algorithm::eltwise_linear, ctx, mkldnn_engine, ctx.GetPlace(), x);

    auto src_memory_p = handler.AcquireSrcMemory(x);
    std::shared_ptr<dnnl::memory> dst_memory_p = nullptr;
    if (is_inplaced) {
      dst_memory_p = src_memory_p;
      out->mutable_data<T>(ctx.GetPlace());
    } else {
      dst_memory_p = handler.AcquireDstMemory(out);
    }
    auto activation_p = handler.AcquireForwardPrimitive();

    auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
    activation_p->execute(astream, {{DNNL_ARG_FROM, *src_memory_p},
                                    {DNNL_ARG_TO, *dst_memory_p}});
    astream.wait();

    out->set_mem_desc(dst_memory_p->get_desc());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(scale, MKLDNN, paddle::platform::CPUPlace,
                   ops::ScaleMKLDNNKernel<float>,
                   ops::ScaleMKLDNNKernel<paddle::platform::bfloat16>);
