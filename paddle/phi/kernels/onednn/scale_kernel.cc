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

#include "paddle/phi/kernels/scale_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& scale,
                 const Scalar& bias,
                 bool bias_after_scale,
                 DenseTensor* out) {
  float alpha = scale.to<float>();
  float beta = bias_after_scale ? bias.to<float>() : bias.to<float>() * alpha;

  funcs::ActivationOneDNNHandler<T> handler(dnnl::algorithm::eltwise_linear,
                                            alpha,
                                            beta,
                                            dev_ctx.GetEngine(),
                                            dev_ctx.GetPlace(),
                                            &x);

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  auto activation_p = handler.AcquireForwardPrimitive();

  bool is_inplaced = x.IsSharedBufferWith(*out);
  std::shared_ptr<dnnl::memory> dst_memory_p = nullptr;
  if (is_inplaced) {
    dst_memory_p = src_memory_p;
    dev_ctx.template Alloc<T>(out);
  } else {
    dst_memory_p = handler.AcquireDstMemory(out);
  }

  auto& astream = OneDNNContext::tls().get_stream();
  activation_p->execute(
      astream, {{DNNL_ARG_FROM, *src_memory_p}, {DNNL_ARG_TO, *dst_memory_p}});
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}

}  // namespace phi

PD_REGISTER_KERNEL(scale,
                   OneDNN,
                   ONEDNN,
                   phi::ScaleKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {}
