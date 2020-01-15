/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <iostream>
#include <numeric>
#include "mkldnn.hpp"
#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::MKLDNNMemDesc;

using mkldnn::memory;  // Note: paddle has also "memory" namespace
using mkldnn::primitive;
using mkldnn::prop_kind;
using mkldnn::softmax_backward;
using mkldnn::softmax_forward;
using mkldnn::stream;
using platform::to_void_cast;

template <typename T>
class SoftmaxMKLDNNHandler
    : public platform::MKLDNNHandlerT<T, mkldnn::softmax_forward,
                                      mkldnn::softmax_backward> {
 public:
  SoftmaxMKLDNNHandler(const std::vector<int64_t>& dims,
                       const MKLDNNMemoryFormat fmt, const int& axis,
                       const platform::MKLDNNDeviceContext& dev_ctx,
                       platform::Place cpu_place, const std::string& uniq_name)
      : platform::MKLDNNHandlerT<T, mkldnn::softmax_forward,
                                 mkldnn::softmax_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(dims, uniq_name)) {
    auto md = mkldnn::memory::desc(dims, platform::MKLDNNGetDataType<T>(), fmt);

    this->AcquireForwardPrimitiveDescriptor(prop_kind::forward_scoring, md,
                                            axis);
  }

  SoftmaxMKLDNNHandler(const std::vector<int64_t>& dims,
                       const MKLDNNMemoryFormat fmt,
                       const MKLDNNMemoryFormat diff_fmt, const int& axis,
                       const platform::MKLDNNDeviceContext& dev_ctx,
                       platform::Place cpu_place, const std::string& uniq_name)
      : platform::MKLDNNHandlerT<T, mkldnn::softmax_forward,
                                 mkldnn::softmax_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(dims, uniq_name)) {
    auto data_softmax_md =
        mkldnn::memory::desc(dims, platform::MKLDNNGetDataType<T>(), fmt);
    auto diff_softmax_md =
        mkldnn::memory::desc(dims, platform::MKLDNNGetDataType<T>(), diff_fmt);

    this->AcquireBackwardPrimitiveDescriptor(diff_softmax_md, data_softmax_md,
                                             axis);
  }
};

template <typename T>
class SoftmaxMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const Tensor* input = ctx.Input<Tensor>("X");
    Tensor* output = ctx.Output<Tensor>("Out");
    PADDLE_ENFORCE_EQ(
        input->dims(), output->dims(),
        "The shape of softmax's input and output must be identical.");

    auto dims = input->dims();  // input and output share the same shape
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), dims.size());

    auto softmax_tz = paddle::framework::vectorize<int64_t>(dims);

    SoftmaxMKLDNNHandler<T> handler(softmax_tz, input->format(), axis, dev_ctx,
                                    ctx.GetPlace(), ctx.OutputName("Out"));

    auto softmax_src_memory_p = handler.AcquireSrcMemory(input);
    auto softmax_dst_memory_p = handler.AcquireDstMemory(output);
    auto softmax_p = handler.AcquireForwardPrimitive();

    mkldnn::stream astream(dev_ctx.GetEngine());
    softmax_p->execute(astream, {{MKLDNN_ARG_SRC, *softmax_src_memory_p},
                                 {MKLDNN_ARG_DST, *softmax_dst_memory_p}});
    astream.wait();

    const bool is_test = ctx.Attr<bool>("is_test");
    if (!is_test) {
      T* output_data = output->mutable_data<T>(ctx.GetPlace());
      std::for_each(output_data, &output_data[output->numel()], [](T& val) {
        val = std::max(val, static_cast<T>(exp(-64)));
      });
    }

    output->set_layout(framework::DataLayout::kMKLDNN);
    // Softmax output format is the same as input one
    output->set_format(input->format());
  }
};

template <typename T>
class SoftmaxMKLDNNGradKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const Tensor* output = ctx.Input<Tensor>("Out");
    auto* dout = ctx.template Input<Tensor>(framework::GradVarName("Out"));
    auto* dx =
        ctx.template Output<framework::Tensor>(framework::GradVarName("X"));

    PADDLE_ENFORCE_EQ(
        dout->dims(), dx->dims(),
        "The shape of softmax_grad's input and output must be identical.");

    auto dims = dout->dims();  // input and output share the same shape
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), dims.size());

    auto softmax_tz = paddle::framework::vectorize<int64_t>(dims);

    SoftmaxMKLDNNHandler<T> handler(softmax_tz, output->format(),
                                    dout->format(), axis, dev_ctx,
                                    ctx.GetPlace(), ctx.InputName("Out"));

    auto dst_memory_p = handler.AcquireDstMemory(output);
    auto diff_dst_memory_p = handler.AcquireDiffDstMemory(dout);
    auto diff_src_memory_p = handler.AcquireDiffSrcMemory(dx);

    auto softmax_bwd_p = handler.AcquireBackwardPrimitive();

    mkldnn::stream astream(dev_ctx.GetEngine());
    softmax_bwd_p->execute(astream,
                           {{MKLDNN_ARG_DST, *dst_memory_p},
                            {MKLDNN_ARG_DIFF_DST, *diff_dst_memory_p},
                            {MKLDNN_ARG_DIFF_SRC, *diff_src_memory_p}});
    astream.wait();

    dx->set_layout(framework::DataLayout::kMKLDNN);
    dx->set_format(dout->format());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(softmax, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::SoftmaxMKLDNNKernel<float>);
REGISTER_OP_KERNEL(softmax_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::SoftmaxMKLDNNGradKernel<float>);
