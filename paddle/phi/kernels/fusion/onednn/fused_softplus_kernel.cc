// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

namespace phi::fusion {

template <typename T, typename Context>
void FusedSoftplusKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         float beta,
                         float threshold UNUSED,
                         const std::string& fuse_activation,
                         const float fuse_alpha,
                         const float fuse_beta,
                         DenseTensor* out) {
  funcs::SoftplusOneDNNHandler<T> handler(
      dev_ctx, &x, beta, fuse_activation, fuse_alpha, fuse_beta);

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  auto beta_memory_p = handler.AcquireBetaMemory(&beta);
  std::shared_ptr<dnnl::memory> dst_memory_p = nullptr;
  if (x.IsSharedBufferWith(*out)) {
    dst_memory_p = src_memory_p;
    dev_ctx.template Alloc<T>(out);
  } else {
    dst_memory_p = handler.AcquireDstMemory(out);
  }
  auto softplus_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();

  const std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC_0, *src_memory_p},
      {DNNL_ARG_SRC_1, *beta_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}};

  softplus_p->execute(astream, args);
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}

}  // namespace phi::fusion

PD_REGISTER_KERNEL(fused_softplus,
                   OneDNN,
                   ONEDNN,
                   phi::fusion::FusedSoftplusKernel,
                   float,
                   phi::dtype::bfloat16) {}
