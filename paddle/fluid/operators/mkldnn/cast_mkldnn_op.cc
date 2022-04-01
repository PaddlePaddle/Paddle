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
class CastMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();

    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    int in_dtype = ctx.Attr<int>("in_dtype");
    int out_dtype = ctx.Attr<int>("out_dtype");

    auto x_paddle_type = framework::proto::VarType::Type(in_dtype);
    auto out_paddle_type = framework::proto::VarType::Type(out_dtype);

    dnnl::memory::data_type x_type = framework::ToMKLDNNDataType(x_paddle_type);
    dnnl::memory::data_type out_type =
        framework::ToMKLDNNDataType(out_paddle_type);

    auto x_tz = phi::vectorize(x->dims());

    platform::ReorderMKLDNNHandler reorder_handler(x_tz, x_paddle_type, x_type,
                                                   out_paddle_type, out_type,
                                                   dev_ctx.GetEngine());

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        x->mem_desc(), platform::to_void_cast(x->data<T>()));
    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
        out, x->mem_desc(), dev_ctx.GetPlace());
    auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                    reorder_src_memory_p);

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
    astream.wait();

    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_mem_desc(reorder_dst_memory_p->get_desc());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(cast, MKLDNN, paddle::platform::CPUPlace,
                   ops::CastMKLDNNKernel<float>,
                   ops::CastMKLDNNKernel<paddle::platform::bfloat16>);
