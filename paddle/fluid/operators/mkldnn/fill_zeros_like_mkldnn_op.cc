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
class FillZerosLikeMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& dnnl_engine = dev_ctx.GetEngine();

    auto* out = ctx.Output<Tensor>("Out");
    const auto out_tz = phi::vectorize(out->dims());

    platform::FillConstantMKLDNNHandler<T> handler(out, dnnl_engine,
                                                   ctx.GetPlace());

    static T zero = static_cast<T>(0);

    dnnl::memory zero_memory =
        dnnl::memory(platform::FillConstantMKLDNNHandler<T>::src1_md,
                     dnnl_engine, reinterpret_cast<uint8_t*>(&zero));

    auto src0_memory_p = handler.AcquireDstMemory(out);
    auto fill_constant_p = handler.AcquireForwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    fill_constant_p->execute(astream, {{DNNL_ARG_SRC_0, *src0_memory_p},
                                       {DNNL_ARG_SRC_1, zero_memory},
                                       {DNNL_ARG_DST, *src0_memory_p}});
    astream.wait();

    out->set_mem_desc({out_tz, platform::MKLDNNGetDataType<T>(),
                       platform::GetPlainMKLDNNFormat(out_tz.size())});
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
