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

#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;

template <typename T>
class Pad3dMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    std::vector<int> paddings(ctx.Attr<std::vector<int>>("paddings"));

    T pad_value = static_cast<T>(ctx.Attr<float>("value"));

    auto x_tz = phi::vectorize(x->dims());
    auto out_tz = phi::vectorize(out->dims());

    auto paddle_dtype = framework::TransToProtoVarType(x->dtype());

    platform::FillConstantMKLDNNHandler<T> handler(out, onednn_engine, ctx.GetPlace());

    dnnl::memory constant_value_memory =
        dnnl::memory(platform::FillConstantMKLDNNHandler<T>::src1_md,
                     onednn_engine,
                     reinterpret_cast<uint8_t*>(&pad_value));

    auto src0_memory_p = handler.AcquireDstMemory(out);
    auto fill_constant_p = handler.AcquireForwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    fill_constant_p->execute(astream,
                             {{DNNL_ARG_SRC_0, *src0_memory_p},
                              {DNNL_ARG_SRC_1, constant_value_memory},
                              {DNNL_ARG_DST, *src0_memory_p}});
    astream.wait();

    // fill_constant handler flattens memory, so we have to revert it now
    const dnnl::memory::desc real_out_md(out_tz, platform::MKLDNNGetDataType<T>(), platform::GetPlainMKLDNNFormat(out_tz.size()));

    platform::ReorderMKLDNNHandler reorder_handler(
      x_tz,
      paddle_dtype,
      framework::ToMKLDNNDataType(paddle_dtype),
      onednn_engine);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(x->mem_desc(), platform::to_void_cast(x->data<T>()));

    auto reorder_dst_memory_p = std::make_shared<dnnl::memory>(real_out_md, onednn_engine, out->data<T>());
    
    std::vector<int64_t> offsets(5, 0); // NCDHW     
    for(int i=0; i<3; ++i) {
      offsets[4-i] = paddings[2*i];
    }
    
    auto slice_mem_p = reorder_handler.AcquireSubmemory(x_tz, offsets, reorder_dst_memory_p);

    auto reorder_p =
        reorder_handler.AcquireReorder(slice_mem_p, reorder_src_memory_p);
    reorder_p->execute(astream, *reorder_src_memory_p, *slice_mem_p);
    astream.wait();

    out->set_mem_desc(real_out_md);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(pad3d, MKLDNN, paddle::platform::CPUPlace,
                   ops::Pad3dMKLDNNKernel<float>,
                   ops::Pad3dMKLDNNKernel<int8_t>,
                   ops::Pad3dMKLDNNKernel<uint8_t>,
                   ops::Pad3dMKLDNNKernel<paddle::platform::bfloat16>);
