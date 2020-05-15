/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using mkldnn::stream;
using mkldnn::sum;

template <typename T>
class EltwiseAddMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const auto* x = ctx.Input<Tensor>("X");
    const auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");

    platform::BinaryMKLDNNHandler<T> handler(
        dev_ctx, mkldnn_engine, ctx.GetPlace(), x, y, z, ctx.OutputName("Out"));

    const auto src_x_memory = handler.AcquireSrcMemory(x);
    const auto src_y_memory = handler.AcquireSecondSrcMemory(y);

    // For Inplace src and and dst are the same memory object
    const auto dst_memory =
        x->IsSharedBufferWith(*z) ? src_x_memory : handler.AcquireDstMemory(z);

    const auto binary_prim = handler.AcquireForwardPrimitive();

    mkldnn::stream astream(mkldnn_engine);

    const std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC_0, *src_x_memory},
        {DNNL_ARG_SRC_1, *src_y_memory},
        {DNNL_ARG_DST, *dst_memory}};

    binary_prim->execute(astream, args);
    astream.wait();

    z->set_layout(DataLayout::kMKLDNN);
    z->set_format(platform::GetMKLDNNFormat(*dst_memory));
  }
};

template <typename T>
class EltwiseAddMKLDNNGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto set_mkldnn_format = [](Tensor* in, const Tensor* out) {
      in->set_layout(DataLayout::kMKLDNN);
      in->set_format(out->format());
    };

    auto blas = math::GetBlas<paddle::platform::CPUDeviceContext, T>(ctx);
    if (dx) {
      blas.VCOPY(dout->numel(), dout->data<T>(),
                 dx->mutable_data<T>(ctx.GetPlace()));
      set_mkldnn_format(dx, dout);
    }

    if (dy) {
      blas.VCOPY(dout->numel(), dout->data<T>(),
                 dy->mutable_data<T>(ctx.GetPlace()));
      set_mkldnn_format(dy, dout);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(elementwise_add, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::EltwiseAddMKLDNNKernel<float>)

REGISTER_OP_KERNEL(elementwise_add_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::EltwiseAddMKLDNNGradKernel<float>)
