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

#include "paddle/phi/kernels/clip_grad_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
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

PD_REGISTER_KERNEL(clip_grad,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::ClipGradKernel,
                   float,
                   phi::dtype::bfloat16) {}
