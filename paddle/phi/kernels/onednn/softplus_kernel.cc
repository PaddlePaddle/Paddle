// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/activation_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
class SoftplusOneDNNHandler : public OneDNNHandlerNoCachingT<T, dnnl::binary> {
 public:
  SoftplusOneDNNHandler(const dnnl::engine onednn_engine,
                        Place cpu_place,
                        const phi::DenseTensor* x,
                        const float beta)
      : OneDNNHandlerNoCachingT<T, dnnl::binary>(onednn_engine, cpu_place) {
    auto x_tz = phi::vectorize(x->dims());

    auto beta_tz = std::vector<int64_t>(x_tz.size(), 1);
    auto beta_md =
        dnnl::memory::desc(beta_tz,
                           platform::OneDNNGetDataType<T>(),
                           platform::GetPlainOneDNNFormat(x_tz.size()));

    dnnl::post_ops post_ops;
    post_ops.append_eltwise(
        1.0f, dnnl::algorithm::eltwise_soft_relu, 0.0f, 0.0f);
    if (beta != 1.0f) {
      post_ops.append_eltwise(
          1.0f, dnnl::algorithm::eltwise_linear, 1.0f / beta, 0.0f);
    }

    platform::AppendActivation(ctx, post_ops);

    dnnl::primitive_attr attrs;
    attrs.set_post_ops(post_ops);

    this->AcquireForwardPrimitiveDescriptor(attrs,
                                            dnnl::algorithm::binary_mul,
                                            x->mem_desc(),
                                            beta_md,
                                            x->mem_desc());
  }

  std::shared_ptr<dnnl::memory> AcquireBetaMemory(const float* beta) {
    return this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->src1_desc(), platform::to_void_cast<float>(beta));
  }
};

template <typename T, typename Context>
void SoftplusKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    float beta,
                    float threshold,
                    DenseTensor* out) {
  SoftplusOneDNNHandler<T> handler(
      dev_ctx.GetEngine(), dev_ctx.GetPlace(), &x, beta);

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  auto beta_memory_p = handler.AcquireBetaMemory(&beta);
  std::shared_ptr<dnnl::memory> dst_memory_p = nullptr;
  if (x.IsSharedBufferWith(*out)) {
    dst_memory_p = src_memory_p;
    dev_ctx.template Alloc<T>(out);
  } else {
    dst_memory_p = handler.AcquireDstMemory(out);
  }
  auto binary_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();

  const std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC_0, *src_memory_p},
      {DNNL_ARG_SRC_1, *beta_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}};

  binary_p->execute(astream, args);
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}

}  // namespace phi

PD_REGISTER_KERNEL(softplus,
                   OneDNN,
                   ONEDNN,
                   phi::SoftplusKernel,
                   float,
                   phi::dtype::bfloat16) {}
