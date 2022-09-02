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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/onednn/onednn_reuse.h"

namespace phi {

namespace funcs {
template <typename T>
class ClipOneDNNHandler
    : public OneDNNHandlerNoCachingT<T,
                                     dnnl::eltwise_forward,
                                     dnnl::eltwise_backward> {
 public:
  ClipOneDNNHandler(const Scalar& min,
                    const Scalar& max,
                    const dnnl::engine engine,
                    Place cpu_place,
                    const DenseTensor* x)
      : OneDNNHandlerNoCachingT<T,
                                dnnl::eltwise_forward,
                                dnnl::eltwise_backward>(engine, cpu_place) {
    float alpha = min.to<float>();
    float beta = max.to<float>();

    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            dnnl::algorithm::eltwise_clip_v2,
                                            x->mem_desc(),
                                            alpha,
                                            beta);
  }

  ClipOneDNNHandler(const Scalar& min,
                    const Scalar& max,
                    const dnnl::engine engine,
                    Place cpu_place,
                    const DenseTensor* x,
                    const DenseTensor* dout)
      : OneDNNHandlerNoCachingT<T,
                                dnnl::eltwise_forward,
                                dnnl::eltwise_backward>(engine, cpu_place) {
    float alpha = min.to<float>();
    float beta = max.to<float>();

    this->AcquireForwardPrimitiveDescriptor(dnnl::prop_kind::forward_training,
                                            dnnl::algorithm::eltwise_clip_v2,
                                            x->mem_desc(),
                                            alpha,
                                            beta);
    this->AcquireBackwardPrimitiveDescriptor(dnnl::algorithm::eltwise_clip_v2,
                                             dout->mem_desc(),
                                             x->mem_desc(),
                                             alpha,
                                             beta);
  }
  std::shared_ptr<dnnl::memory> AcquireBackwardSrcMemory(
      const DenseTensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(this->bwd_pd_->src_desc(),
                                            to_void_cast<T>(input_data));
  }
};

}  // namespace funcs

template <typename T, typename Context>
void ClipKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const Scalar& min,
                const Scalar& max,
                DenseTensor* out) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  funcs::ClipOneDNNHandler<T> handler(
      min, max, onednn_engine, dev_ctx.GetPlace(), &x);

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  auto dst_memory_p = handler.AcquireDstMemory(out);
  auto activation_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  activation_p->execute(
      astream, {{DNNL_ARG_FROM, *src_memory_p}, {DNNL_ARG_TO, *dst_memory_p}});
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}

template <typename T, typename Context>
void ClipGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    const Scalar& min,
                    const Scalar& max,
                    DenseTensor* x_grad) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  funcs::ClipOneDNNHandler<T> handler(
      min, max, onednn_engine, dev_ctx.GetPlace(), &x, &out_grad);

  auto src_memory_p = handler.AcquireBackwardSrcMemory(&x);
  auto diff_dst_memory_p = handler.AcquireDiffDstMemory(&out_grad);
  auto diff_src_memory_p = handler.AcquireDiffSrcMemory(x_grad);
  auto activation_backward_p = handler.AcquireBackwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  activation_backward_p->execute(astream,
                                 {{DNNL_ARG_SRC, *src_memory_p},
                                  {DNNL_ARG_DIFF_DST, *diff_dst_memory_p},
                                  {DNNL_ARG_DIFF_SRC, *diff_src_memory_p}});
  astream.wait();

  x_grad->set_mem_desc(diff_dst_memory_p->get_desc());
}

}  // namespace phi

PD_REGISTER_KERNEL(
    clip, OneDNN, ALL_LAYOUT, phi::ClipKernel, float, phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(clip_grad,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::ClipGradKernel,
                   float,
                   phi::dtype::bfloat16) {}
