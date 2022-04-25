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

#include "paddle/fluid/framework/expect.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using dnnl::memory;
using framework::Tensor;
using platform::GetMKLDNNFormat;
using platform::MKLDNNDeviceContext;
using platform::MKLDNNGetDataType;
using platform::to_void_cast;

namespace {
template <typename T>
class PReluMKLDNNHandler
    : public platform::MKLDNNHandlerT<T, dnnl::prelu_forward,
                                      dnnl::prelu_backward> {
 public:
  PReluMKLDNNHandler(const MKLDNNDeviceContext& dev_ctx,
                     const dnnl::engine engine, platform::Place cpu_place,
                     const Tensor* x, const Tensor* weights,
                     const std::string& uniq_name, const std::string& mode,
                     const std::string& data_format, bool is_test = false)
      : platform::MKLDNNHandlerT<T, dnnl::prelu_forward, dnnl::prelu_backward>(
            dev_ctx, engine, cpu_place,
            platform::CreateKey(dev_ctx, phi::vectorize(x->dims()),
                                uniq_name)) {
    if (unlikely(!this->isCached())) {
      auto weights_dims = phi::vectorize(weights->dims());

      // weights must have same size as X only for "element" case
      if (weights->dims().size() != x->dims().size()) {
        auto new_weights_dims = std::vector<int64_t>(x->dims().size(), 1);
        if (mode == "channel") {
          new_weights_dims[1] =
              *std::max_element(weights_dims.begin(), weights_dims.end());
        }
        weights_dims = std::move(new_weights_dims);
      }
      auto weights_md = memory::desc(weights_dims, MKLDNNGetDataType<T>(),
                                     memory::format_tag::any);

      this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                              x->mem_desc(), weights_md);
      if (!is_test)
        this->AcquireBackwardPrimitiveDescriptor(x->mem_desc(), weights_md,
                                                 x->mem_desc(), weights_md);
    }
  }

  std::shared_ptr<memory> AcquireWeightsMemoryPossiblyWithReorder(
      const Tensor* weights, const bool is_test) {
    const T* weights_data = weights->data<T>();

    // if weights are 1D, every format tag is correct, so we accept
    // format_tag::any's output and no reorder is needed
    if (weights->dims().size() == 1) {
      return this->AcquireMemoryFromPrimitive(this->fwd_pd_->weights_desc(),
                                              to_void_cast<T>(weights_data),
                                              "@alpha_mem_p");
    }

    return this->AcquireMemoryWithReorder(
        weights->mem_desc(), this->fwd_pd_->weights_desc(),
        to_void_cast<T>(weights_data), "@alpha_mem_p", is_test);
  }

  std::shared_ptr<memory> AcquireDiffWeightsMemory(Tensor* output) {
    T* output_data = output->mutable_data<T>(
        this->place_, this->bwd_pd_->diff_weights_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->bwd_pd_->diff_weights_desc(),
                                            output_data, "@diff_weights_mem_p");
  }
};
}  // anonymous namespace

template <typename T>
class PReluMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    const auto* x = ctx.Input<Tensor>("X");
    const auto* alpha = ctx.Input<Tensor>("Alpha");
    auto* out = ctx.Output<Tensor>("Out");
    const bool is_test = ctx.Attr<bool>("is_test");
    const auto mode = ctx.Attr<std::string>("mode");
    const auto data_format = ctx.Attr<std::string>("data_format");

    PReluMKLDNNHandler<T> handler(dev_ctx, onednn_engine, ctx.GetPlace(), x,
                                  alpha, ctx.InputName("X"), mode, data_format,
                                  is_test);

    auto src_memory_p = handler.AcquireSrcMemory(x);
    auto weights_memory_p =
        handler.AcquireWeightsMemoryPossiblyWithReorder(alpha, is_test);
    auto dst_memory_p = handler.AcquireDstMemory(out);
    auto prelu_p = handler.AcquireForwardPrimitive();

    auto& astream = MKLDNNDeviceContext::tls().get_stream();
    prelu_p->execute(astream, {{DNNL_ARG_SRC, *src_memory_p},
                               {DNNL_ARG_WEIGHTS, *weights_memory_p},
                               {DNNL_ARG_DST, *dst_memory_p}});
    astream.wait();

    out->set_mem_desc(dst_memory_p->get_desc());
  }
};

template <typename T>
class PReluGradMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("X");
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dalpha = ctx.Output<Tensor>(framework::GradVarName("Alpha"));
    auto* alpha = ctx.Input<Tensor>("Alpha");
    const bool is_test = ctx.Attr<bool>("is_test");
    const auto mode = ctx.Attr<std::string>("mode");
    const auto data_format = ctx.Attr<std::string>("data_format");

    PReluMKLDNNHandler<T> handler(dev_ctx, onednn_engine, ctx.GetPlace(), x,
                                  alpha, framework::GradVarName("X"), mode,
                                  data_format);

    auto src_memory_p = handler.AcquireSrcMemory(x);
    auto weights_memory_p =
        handler.AcquireWeightsMemoryPossiblyWithReorder(alpha, is_test);
    auto diff_src_memory_p = handler.AcquireDiffSrcMemory(dx);
    auto diff_weights_memory_p = handler.AcquireDiffWeightsMemory(dalpha);
    auto diff_dst_memory_p = handler.AcquireDiffDstMemory(dout);
    auto prelu_p = handler.AcquireBackwardPrimitive();

    auto& astream = MKLDNNDeviceContext::tls().get_stream();
    prelu_p->execute(astream,
                     {{DNNL_ARG_SRC, *src_memory_p},
                      {DNNL_ARG_WEIGHTS, *weights_memory_p},
                      {DNNL_ARG_DIFF_DST, *diff_dst_memory_p},
                      {DNNL_ARG_DIFF_SRC, *diff_src_memory_p},
                      {DNNL_ARG_DIFF_WEIGHTS, *diff_weights_memory_p}});
    astream.wait();

    dx->set_mem_desc(diff_src_memory_p->get_desc());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(prelu, MKLDNN, paddle::platform::CPUPlace,
                   ops::PReluMKLDNNKernel<float>,
                   ops::PReluMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(prelu_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::PReluGradMKLDNNKernel<float>,
                   ops::PReluGradMKLDNNKernel<paddle::platform::bfloat16>);
