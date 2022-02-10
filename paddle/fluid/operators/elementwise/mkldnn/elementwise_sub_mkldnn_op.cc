
// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/elementwise/mkldnn/elementwise_mkldnn_op.h"
namespace paddle {
namespace framework {
class ExecutionContext;
}  // namespace framework
namespace platform {
class CPUDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {
template <typename T>
class EltwiseSubMKLDNNGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto tz = framework::vectorize<int64_t>(dout->dims());
    memory::data_type dout_type = framework::ToMKLDNNDataType(dout->type());
    platform::ReorderMKLDNNHandler handler(tz, dout->type(), dout_type,
                                           onednn_engine);

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    auto reorder_src_memory_p = handler.AcquireSrcMemory(
        dout->format(), platform::to_void_cast(dout->data<T>()));

    if (dx) {
      auto reorder_dst_memory_p =
          handler.AcquireDstMemory(dx, dout->format(), ctx.GetPlace());
      auto reorder_p =
          handler.AcquireReorder(reorder_dst_memory_p, reorder_src_memory_p);
      platform::RecordEvent record_reorder("int_reorder",
                                           platform::EventRole::kUniqueOp);

      reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
      astream.wait();

      dx->set_layout(DataLayout::kMKLDNN);
      dx->set_format(platform::GetMKLDNNFormat(*reorder_dst_memory_p));
    }

    if (dy) {
      // Direct copy
      if (dout->dims() == dy->dims()) {
        auto reorder_dst_memory_p =
            handler.AcquireDstMemory(dy, dout->format(), ctx.GetPlace());

        dnnl::primitive_attr reorder_attr;
        std::vector<float> scales = {-1};
        reorder_attr.set_output_scales(0, scales);
        auto reorder_p = std::make_shared<dnnl::reorder>(
            *(reorder_src_memory_p), *(reorder_dst_memory_p), reorder_attr);
        platform::RecordEvent record_reorder("int_reorder",
                                             platform::EventRole::kUniqueOp);
        reorder_p->execute(astream, *reorder_src_memory_p,
                           *reorder_dst_memory_p);
        astream.wait();

        dy->set_layout(DataLayout::kMKLDNN);
        dy->set_format(platform::GetMKLDNNFormat(*reorder_dst_memory_p));
      } else {
        // Broadcasting

        dnnl::post_ops po;
        po.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, -1.0f, 0);
        dnnl::primitive_attr attr;
        attr.set_post_ops(po);

        platform::ReductionMKLDNNHandler<T> handler_sum(
            dnnl::algorithm::reduction_sum, 0.0f, 0.0f, onednn_engine,
            ctx.GetPlace(), dout, dy, CalculateBroadcastedDims(dout, dy), attr);

        auto dy_memory_p = handler_sum.AcquireDstMemory(dy);
        auto reduction_p = handler_sum.AcquireForwardPrimitive();

        reduction_p->execute(astream, {
                                          {DNNL_ARG_SRC, *reorder_src_memory_p},
                                          {DNNL_ARG_DST, *dy_memory_p},
                                      });
        astream.wait();

        dy->set_layout(DataLayout::kMKLDNN);
        dy->set_format(
            platform::GetMKLDNNFormat(dy_memory_p->get_desc().reshape(
                paddle::framework::vectorize<int64_t>(dy->dims()))));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(
    elementwise_sub, MKLDNN, paddle::platform::CPUPlace,
    ops::EltwiseMKLDNNKernel<float, dnnl::algorithm::binary_sub>,
    ops::EltwiseMKLDNNKernel<paddle::platform::bfloat16,
                             dnnl::algorithm::binary_sub>,
    ops::EltwiseMKLDNNKernel<int8_t, dnnl::algorithm::binary_sub>,
    ops::EltwiseMKLDNNKernel<uint8_t, dnnl::algorithm::binary_sub>)

REGISTER_OP_KERNEL(elementwise_sub_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::EltwiseSubMKLDNNGradKernel<paddle::platform::bfloat16>,
                   ops::EltwiseSubMKLDNNGradKernel<float>)
