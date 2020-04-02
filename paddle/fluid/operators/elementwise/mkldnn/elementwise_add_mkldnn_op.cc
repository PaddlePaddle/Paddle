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
using mkldnn::reorder;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::sum;

template <typename T>
class EltwiseAddMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");

    PADDLE_ENFORCE_EQ(
        x->layout(), DataLayout::kMKLDNN,
        platform::errors::InvalidArgument("Wrong layout set for X tensor"));
    PADDLE_ENFORCE_NE(
        x->format(), MKLDNNMemoryFormat::undef,
        platform::errors::InvalidArgument("Wrong format set for X tensor"));

    PADDLE_ENFORCE_EQ(
        y->layout(), DataLayout::kMKLDNN,
        platform::errors::InvalidArgument("Wrong layout set for Y tensor"));
    PADDLE_ENFORCE_NE(
        y->format(), MKLDNNMemoryFormat::undef,
        platform::errors::InvalidArgument("Wrong format set for Y tensor"));

    auto src_x_tz = framework::vectorize<int64_t>(x->dims());
    auto src_y_tz = framework::vectorize<int64_t>(y->dims());
    auto dst_tz = framework::vectorize<int64_t>(z->dims());

    // Currently MKL-DNN kernel supports only Z <- X + Y, shape(X) == shape(Y)
    // TODO(jczaja): Binary primitive support broadcasting, so we can support
    // this in kernel
    platform::BinaryMKLDNNHandler<T> handler(
        dnnl::algorithm::binary_add, src_x_tz, x->format(), y->format(),
        dev_ctx, ctx.GetPlace(), ctx.OutputName("Out"));

    auto src_x_memory = handler.AcquireSrcMemory(x);
    auto src_y_memory = handler.AcquireSecondSrcMemory(y);

    // For Inplace src and and dst are the same memory object
    auto dst_memory =
        x->IsSharedBufferWith(*z) ? src_x_memory : handler.AcquireDstMemory(z);

    auto binary_prim = handler.AcquireForwardPrimitive();

    mkldnn::stream astream(mkldnn_engine);

    std::unordered_map<int, dnnl::memory> args = {
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
