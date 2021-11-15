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

using framework::Tensor;

template <typename T>
class FillZerosLikeMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T, dnnl::eltwise_forward,
                                               dnnl::eltwise_backward> {
 public:
  FillZerosLikeMKLDNNHandler(Tensor* out, const dnnl::engine engine,
                             platform::Place cpu_place)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::eltwise_forward,
                                          dnnl::eltwise_backward>(engine,
                                                                  cpu_place) {
    // uint8_t is always used to treat float NaNs like normal numbers
    auto md = dnnl::memory::desc(
        {out->numel() * static_cast<int64_t>(sizeof(T))},
        dnnl::memory::data_type::u8, dnnl::memory::format_tag::a);

    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            dnnl::algorithm::eltwise_linear, md,
                                            0.0f, 0.0f);
  }
};

template <typename T>
class FillZerosLikeMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& dnnl_engine = dev_ctx.GetEngine();

    auto* out = ctx.Output<Tensor>("Out");

    FillZerosLikeMKLDNNHandler<T> handler(out, dnnl_engine, ctx.GetPlace());

    auto dst_memory_p = handler.AcquireDstMemory(out);
    auto activation_p = handler.AcquireForwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    activation_p->execute(astream, {{DNNL_ARG_SRC, *dst_memory_p},
                                    {DNNL_ARG_DST, *dst_memory_p}});
    astream.wait();

    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_format(platform::GetPlainMKLDNNFormat(out->dims().size()));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(fill_zeros_like, MKLDNN, paddle::platform::CPUPlace,
                   ops::FillZerosLikeMKLDNNKernel<float>,
                   ops::FillZerosLikeMKLDNNKernel<paddle::platform::bfloat16>,
                   ops::FillZerosLikeMKLDNNKernel<int8_t>,
                   ops::FillZerosLikeMKLDNNKernel<uint8_t>);

REGISTER_OP_KERNEL(fill_zeros_like2, MKLDNN, paddle::platform::CPUPlace,
                   ops::FillZerosLikeMKLDNNKernel<float>,
                   ops::FillZerosLikeMKLDNNKernel<paddle::platform::bfloat16>,
                   ops::FillZerosLikeMKLDNNKernel<int8_t>,
                   ops::FillZerosLikeMKLDNNKernel<uint8_t>);
