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
  SoftmaxMKLDNNHandler(const std::vector<int>& dims,
                       const MKLDNNMemoryFormat fmt,
                       const platform::MKLDNNDeviceContext& dev_ctx,
                       platform::Place cpu_place, const std::string& uniq_name)
      : platform::MKLDNNHandlerT<T, mkldnn::softmax_forward,
                                 mkldnn::softmax_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(dims, uniq_name)) {
    auto md = mkldnn::memory::desc(dims, platform::MKLDNNGetDataType<T>(), fmt);

    this->AcquireForwardPrimitiveDescriptor(prop_kind::forward_scoring, md,
                                            1 /*dim: C*/);
  }

  SoftmaxMKLDNNHandler(const std::vector<int>& dims,
                       const MKLDNNMemoryFormat fmt,
                       const MKLDNNMemoryFormat diff_fmt,
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

    this->AcquireForwardPrimitiveDescriptor(prop_kind::forward_scoring,
                                            data_softmax_md, 1 /*dim: C*/);
    this->AcquireBackwardPrimitiveDescriptor(diff_softmax_md, data_softmax_md,
                                             1 /* dim: C*/);
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

    // flatten input and output to 2-D matrixs
    auto dims = input->dims();  // input and output share the same shape
    auto flattened_dims = framework::flatten_to_2d(dims, dims.size() - 1);

    auto src_tz = paddle::framework::vectorize<int>(flattened_dims);
    auto dst_tz = src_tz;
    // Same memory descriptor to be used for input and output
    memory::dims softmax_tz = {src_tz[0], src_tz[1]};

    SoftmaxMKLDNNHandler<T> handler(softmax_tz, MKLDNNMemoryFormat::nc, dev_ctx,
                                    ctx.GetPlace(), ctx.op().Output("Out"));
    // Currently only NC data format is supported
    auto softmax_src_memory_p = handler.AcquireSrcMemory(input);
    auto softmax_dst_memory_p = handler.AcquireDstMemory(output);
    auto softmax_p = handler.AcquireForwardPrimitive(*softmax_src_memory_p,
                                                     *softmax_dst_memory_p);

    std::vector<primitive> pipeline{*softmax_p};
    stream(stream::kind::eager).submit(pipeline).wait();

    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    const bool is_test = ctx.Attr<bool>("is_test");
    if (!is_test) {
      T threshold = exp(-64);
      for (int i = 0; i < dst_tz[0] * dst_tz[1]; ++i) {
        output_data[i] =
            output_data[i] < threshold ? threshold : output_data[i];
      }
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
    auto flattened_dims = framework::flatten_to_2d(dims, dims.size() - 1);

    std::vector<int> dst_tz = paddle::framework::vectorize<int>(flattened_dims);
    std::vector<int> src_tz(dst_tz);

    // Same memory descriptor to be used for input and output
    memory::dims softmax_tz = {src_tz[0], src_tz[1]};

    // TODO(jczaja): Add layouts support when there is a need to do so
    // Two dimensional softmax does support NC format
    // Normalization is made after innermost dimension eg. C out of NC
    SoftmaxMKLDNNHandler<T> handler(softmax_tz, MKLDNNMemoryFormat::nc,
                                    MKLDNNMemoryFormat::nc, dev_ctx,
                                    ctx.GetPlace(), ctx.op().Input("Out"));

    auto dst_memory_p = handler.AcquireDstMemory(output);
    auto diff_dst_memory_p = handler.AcquireDiffDstMemory(dout);
    auto diff_src_memory_p = handler.AcquireDiffSrcMemory(dx);

    // Get primitve from device context
    auto softmax_bwd_p = handler.AcquireBackwardPrimitive(
        *dst_memory_p, *diff_dst_memory_p, *diff_src_memory_p);

    std::vector<primitive> pipeline{*softmax_bwd_p};
    stream(stream::kind::eager).submit(pipeline).wait();
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(softmax, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::SoftmaxMKLDNNKernel<float>);
REGISTER_OP_KERNEL(softmax_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::SoftmaxMKLDNNGradKernel<float>);
