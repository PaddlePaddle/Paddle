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

#include "paddle/phi/kernels/softmax_grad_kernel.h"

#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SoftmaxGradKernel(const Context& dev_ctx,
                       const DenseTensor& out,
                       const DenseTensor& out_grad,
                       int axis,
                       DenseTensor* x_grad) {
  funcs::SoftmaxOneDNNHandler<T> handler(
      dev_ctx.GetEngine(), dev_ctx.GetPlace(), axis, &out, &out_grad);

  auto dst_memory_p = handler.AcquireDstMemory(&out);
  auto diff_dst_memory_p = handler.AcquireDiffDstMemory(&out_grad);
  auto diff_src_memory_p = handler.AcquireDiffSrcMemory(x_grad);

  auto softmax_bwd_p = handler.AcquireBackwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  softmax_bwd_p->execute(astream,
                         {{DNNL_ARG_DST, *dst_memory_p},
                          {DNNL_ARG_DIFF_DST, *diff_dst_memory_p},
                          {DNNL_ARG_DIFF_SRC, *diff_src_memory_p}});
  astream.wait();

  x_grad->set_mem_desc(diff_src_memory_p->get_desc());
}

}  // namespace phi

PD_REGISTER_KERNEL(
    softmax_grad, OneDNN, ALL_LAYOUT, phi::SoftmaxGradKernel, float) {}
