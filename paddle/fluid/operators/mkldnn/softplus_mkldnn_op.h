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

#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;

template <typename T>
class SoftplusMKLDNNHandler
    : public platform::MKLDNNHandlerNoCachingT<T, dnnl::binary> {
 public:
  SoftplusMKLDNNHandler(const Tensor* x, const float beta,
                        const mkldnn::engine engine, platform::Place cpu_place)
      : platform::MKLDNNHandlerNoCachingT<T, dnnl::binary>(engine, cpu_place) {
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

    dnnl::primitive_attr attrs;
    attrs.set_post_ops(post_ops);

    this->AcquireForwardPrimitiveDescriptor(attrs, dnnl::algorithm::binary_mul,
                                            x_md, beta_md, x_md);
  }

  std::shared_ptr<mkldnn::memory> AcquireBetaMemory(const float* beta) {
    return this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->src1_desc(), platform::to_void_cast<float>(beta));
  }
};

template <typename T>
void custom_softplus_eltwise_forward(const framework::ExecutionContext& ctx) {
  const auto& dev_ctx =
      ctx.template device_context<platform::MKLDNNDeviceContext>();
  const auto& mkldnn_engine = dev_ctx.GetEngine();

  const auto* x = ctx.Input<Tensor>("X");
  auto* out = ctx.Output<Tensor>("Out");

  bool is_inplaced = x->IsSharedBufferWith(*out);

  const float beta = ctx.Attr<float>("beta");

  SoftplusMKLDNNHandler<T> handler(x, beta, mkldnn_engine, ctx.GetPlace());

  auto src_memory_p = handler.AcquireSrcMemory(x);

  auto beta_memory_p = handler.AcquireBetaMemory(&beta);
  auto dst_memory_p =
      is_inplaced ? src_memory_p : handler.AcquireDstMemory(out);
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
