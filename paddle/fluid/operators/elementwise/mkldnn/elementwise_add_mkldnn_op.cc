// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
struct CPUPlace;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {
template <typename T>
class EltwiseAddMKLDNNGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto tz = paddle::framework::vectorize<int64_t>(dout->dims());
    memory::data_type dout_type = framework::ToMKLDNNDataType(dout->type());
    std::string key = platform::CreateKey(dev_ctx, tz, dout->format(),
                                          dout->format(), dout_type);
    platform::ReorderMKLDNNHandler handler(tz, dout->type(), dout_type, dev_ctx,
                                           onednn_engine, key);

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
    }

    if (dy) {
      auto reorder_dst_memory_p =
          handler.AcquireDstMemory(dy, dout->format(), ctx.GetPlace());
      auto reorder_p =
          handler.AcquireReorder(reorder_dst_memory_p, reorder_src_memory_p);
      platform::RecordEvent record_reorder("int_reorder",
                                           platform::EventRole::kUniqueOp);
      reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
      astream.wait();
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(
    elementwise_add, MKLDNN, ::paddle::platform::CPUPlace,
    ops::EltwiseMKLDNNKernel<float, dnnl::algorithm::binary_add>,
    ops::EltwiseMKLDNNKernel<paddle::platform::bfloat16,
                             dnnl::algorithm::binary_add>,
    ops::EltwiseMKLDNNKernel<int8_t, dnnl::algorithm::binary_add>,
    ops::EltwiseMKLDNNKernel<uint8_t, dnnl::algorithm::binary_add>)

REGISTER_OP_KERNEL(elementwise_add_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::EltwiseAddMKLDNNGradKernel<paddle::platform::bfloat16>,
                   ops::EltwiseAddMKLDNNGradKernel<float>)
