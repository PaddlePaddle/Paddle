/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

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
class EltwiseMulMKLDNNGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    if (dx) {
      // dx = dout*y
      platform::BinaryMKLDNNHandler<T> handler(
          dnnl::algorithm::binary_mul, axis, mkldnn_engine, ctx.GetPlace(),
          dout, y, dx, 1.0f, 1.0f, 1.0f);

      const auto src_dout_memory = handler.AcquireSrcMemory(dout);
      const auto src_y_memory = handler.AcquireSecondSrcMemory(y);
      const auto dst_dx_memory = handler.AcquireDstMemory(dx);

      const auto binary_prim = handler.AcquireForwardPrimitive();

      const std::unordered_map<int, dnnl::memory> args = {
          {DNNL_ARG_SRC_0, *src_dout_memory},
          {DNNL_ARG_SRC_1, *src_y_memory},
          {DNNL_ARG_DST, *dst_dx_memory}};

      binary_prim->execute(astream, args);
      astream.wait();

      dx->set_layout(framework::DataLayout::kMKLDNN);
      dx->set_format(platform::GetMKLDNNFormat(*dst_dx_memory));
    }

    if (dy) {
      // dy = dout*x
      // Handler is having nullptr passed instead of output tensor as
      // we want Dst buffer to be allocated by oneDNN not to use Tensor
      platform::BinaryMKLDNNHandler<T> handler(
          dnnl::algorithm::binary_mul, axis, mkldnn_engine, ctx.GetPlace(),
          dout, x, nullptr, 1.0f, 1.0f, 1.0f);

      const auto src_dout_memory = handler.AcquireSrcMemory(dout);
      const auto src_x_memory = handler.AcquireSecondSrcMemory(x);

      // If broadcasting is in use then let's write to temporary
      // buffer allocated by oneDNN
      const auto dst_dy_memory = (dout->dims() == dy->dims())
                                     ? handler.AcquireDstMemory(dy)
                                     : handler.AcquireDstMemory();

      const auto binary_prim = handler.AcquireForwardPrimitive();

      const std::unordered_map<int, dnnl::memory> args = {
          {DNNL_ARG_SRC_0, *src_dout_memory},
          {DNNL_ARG_SRC_1, *src_x_memory},
          {DNNL_ARG_DST, *dst_dy_memory}};

      binary_prim->execute(astream, args);
      astream.wait();

      dy->set_layout(framework::DataLayout::kMKLDNN);

      // Reduction is needed for broadcasting scenario
      if (dout->dims() != dy->dims()) {
        platform::ReductionMKLDNNHandler<T> handler_sum(
            dnnl::algorithm::reduction_sum, 0.0f, 0.0f, mkldnn_engine,
            ctx.GetPlace(), dout, dy, CalculateBroadcastedDims(dout, dy));
        auto dy_memory_p = handler_sum.AcquireDstMemory(dy);
        auto reduction_p = handler_sum.AcquireForwardPrimitive();
        // As source we use mem object with results from binary operation
        reduction_p->execute(astream, {{DNNL_ARG_SRC, *dst_dy_memory},
                                       {DNNL_ARG_DST, *dy_memory_p}});
        astream.wait();
        dy->set_format(
            platform::GetMKLDNNFormat(dy_memory_p->get_desc().reshape(
                paddle::framework::vectorize<int64_t>(dy->dims()))));

      } else {
        dy->set_format(platform::GetMKLDNNFormat(*dst_dy_memory));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(
    elementwise_mul, MKLDNN, ::paddle::platform::CPUPlace,
    ops::EltwiseMKLDNNKernel<float, dnnl::algorithm::binary_mul>,
    ops::EltwiseMKLDNNKernel<paddle::platform::bfloat16,
                             dnnl::algorithm::binary_mul>,
    ops::EltwiseMKLDNNKernel<int8_t, dnnl::algorithm::binary_mul>,
    ops::EltwiseMKLDNNKernel<uint8_t, dnnl::algorithm::binary_mul>)

REGISTER_OP_KERNEL(elementwise_mul_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::EltwiseMulMKLDNNGradKernel<paddle::platform::bfloat16>,
                   ops::EltwiseMulMKLDNNGradKernel<float>)
