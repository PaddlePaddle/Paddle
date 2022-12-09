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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"

namespace paddle {
namespace operators {

template <typename T>
class ShuffleChannelMKLDNNHandler
    : public phi::funcs::OneDNNHandlerNoCachingT<T, dnnl::shuffle_forward> {
 public:
  ShuffleChannelMKLDNNHandler(const phi::DenseTensor* x,
                              const int group,
                              const dnnl::engine engine,
                              platform::Place cpu_place)
      : phi::funcs::OneDNNHandlerNoCachingT<T, dnnl::shuffle_forward>(
            engine, cpu_place) {
    static constexpr int channel_axis = 1;
    this->AcquireForwardPrimitiveDescriptor(
        dnnl::prop_kind::forward_training, x->mem_desc(), channel_axis, group);
  }
};

template <typename T>
class ShuffleChannelMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto& dev_ctx = ctx.template device_context<phi::OneDNNContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    const auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");

    // oneDNN handles group using C/g instead of g
    const int group = x->dims()[1] / ctx.Attr<int>("group");

    ShuffleChannelMKLDNNHandler<T> handler(
        x, group, onednn_engine, ctx.GetPlace());

    auto src_memory_p = handler.AcquireSrcMemory(x);
    auto dst_memory_p = handler.AcquireDstMemory(out);

    auto shuffle_p = handler.AcquireForwardPrimitive();

    auto& astream = phi::OneDNNContext::tls().get_stream();
    shuffle_p->execute(
        astream,
        {{DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}});
    astream.wait();

    out->set_mem_desc(dst_memory_p->get_desc());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(shuffle_channel,
                   MKLDNN,
                   phi::CPUPlace,
                   ops::ShuffleChannelMKLDNNKernel<float>,
                   ops::ShuffleChannelMKLDNNKernel<paddle::platform::bfloat16>);
