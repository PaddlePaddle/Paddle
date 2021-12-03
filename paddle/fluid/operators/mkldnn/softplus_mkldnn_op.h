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

#pragma once
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;

template <typename T>
class SoftplusMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T, dnnl::binary> {
 public:
  SoftplusMKLDNNHandler(const framework::ExecutionContext& ctx, const Tensor* x,
                        const float beta, const dnnl::engine engine)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::binary>(engine,
                                                           ctx.GetPlace()) {
    auto x_tz = framework::vectorize(x->dims());
    auto x_md =
        dnnl::memory::desc(x_tz, platform::MKLDNNGetDataType<T>(), x->format());

    auto beta_tz = std::vector<int64_t>(x_tz.size(), 1);
    auto beta_md = dnnl::memory::desc(beta_tz, platform::MKLDNNGetDataType<T>(),
                                      x->format());

    dnnl::post_ops post_ops;
    post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_soft_relu, 0.0f,
                            0.0f);
    if (beta != 1.0f) {
      post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear,
                              1.0f / beta, 0.0f);
    }

    AppendFusedActivationIfExists(ctx, &post_ops);

    dnnl::primitive_attr attrs;
    attrs.set_post_ops(post_ops);

    this->AcquireForwardPrimitiveDescriptor(attrs, dnnl::algorithm::binary_mul,
                                            x_md, beta_md, x_md);
  }

  std::shared_ptr<dnnl::memory> AcquireBetaMemory(const float* beta) {
    return this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->src1_desc(), platform::to_void_cast<float>(beta));
  }

 private:
  void AppendFusedActivationIfExists(const framework::ExecutionContext& ctx,
                                     dnnl::post_ops* post_ops) {
    const auto& fused_activation_type =
        algo_map.find(ctx.Attr<std::string>("fuse_activation_type"));

    if (fused_activation_type != algo_map.end()) {
      auto scale_out =
          ctx.Attr<float>("fuse_activation_scale");  // for future int8 support
      post_ops->append_eltwise(scale_out, fused_activation_type->second,
                               ctx.Attr<float>("fuse_activation_alpha"),
                               ctx.Attr<float>("fuse_activation_beta"));
    }
  }

  static const std::unordered_map<std::string, dnnl::algorithm> algo_map;
};

template <typename T>
const std::unordered_map<std::string, dnnl::algorithm>
    SoftplusMKLDNNHandler<T>::algo_map = {
        {"relu", dnnl::algorithm::eltwise_relu},
        {"tanh", dnnl::algorithm::eltwise_tanh},
        {"leaky_relu", dnnl::algorithm::eltwise_relu},
        {"swish", dnnl::algorithm::eltwise_swish},
        {"hardswish", dnnl::algorithm::eltwise_hardswish},
        {"sqrt", dnnl::algorithm::eltwise_sqrt},
        {"abs", dnnl::algorithm::eltwise_abs},
        {"clip", dnnl::algorithm::eltwise_clip},
        {"gelu", dnnl::algorithm::eltwise_gelu_erf},
        {"gelu_tanh", dnnl::algorithm::eltwise_gelu_tanh},
        {"relu6", dnnl::algorithm::eltwise_bounded_relu},
        {"sigmoid", dnnl::algorithm::eltwise_logistic}};

template <typename T>
void custom_softplus_eltwise_forward(const framework::ExecutionContext& ctx) {
  const auto& dev_ctx =
      ctx.template device_context<platform::MKLDNNDeviceContext>();
  const auto& mkldnn_engine = dev_ctx.GetEngine();

  const auto* x = ctx.Input<Tensor>("X");
  auto* out = ctx.Output<Tensor>("Out");

  bool is_inplaced = x->IsSharedBufferWith(*out);

  const float beta = ctx.Attr<float>("beta");

  SoftplusMKLDNNHandler<T> handler(ctx, x, beta, mkldnn_engine);

  auto src_memory_p = handler.AcquireSrcMemory(x);

  auto beta_memory_p = handler.AcquireBetaMemory(&beta);
  std::shared_ptr<dnnl::memory> dst_memory_p = nullptr;
  if (is_inplaced) {
    dst_memory_p = src_memory_p;
    out->mutable_data<T>(ctx.GetPlace());
  } else {
    dst_memory_p = handler.AcquireDstMemory(out);
  }
  auto binary_p = handler.AcquireForwardPrimitive();

  auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();

  const std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC_0, *src_memory_p},
      {DNNL_ARG_SRC_1, *beta_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}};

  binary_p->execute(astream, args);
  astream.wait();

  out->set_layout(framework::DataLayout::kMKLDNN);
  out->set_format(platform::GetMKLDNNFormat(*dst_memory_p));
}
}  // namespace operators
}  // namespace paddle
