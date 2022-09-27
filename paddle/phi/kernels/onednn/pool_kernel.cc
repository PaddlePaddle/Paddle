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

#include "paddle/phi/kernels/pool_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void Pool2dKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const IntArray& kernel_size,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool ceil_mode,
                  bool exclusive,
                  const std::string& data_format,
                  const std::string& pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string& padding_algorithm,
                  DenseTensor* out) {
  funcs::PoolingOneDNNHandler<T> handler(pooling_type,
                                         kernel_size,
                                         strides,
                                         paddings,
                                         global_pooling,
                                         padding_algorithm,
                                         ceil_mode,
                                         exclusive,
                                         adaptive,
                                         dev_ctx.GetEngine(),
                                         dev_ctx.GetPlace(),
                                         &x,
                                         out);

  auto src_memory = handler.AcquireSrcMemory(&x);
  auto dst_memory = handler.AcquireDstMemory(out);

  auto pool_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  if (pooling_type == "max") {
    // Training
    auto workspace_memory = handler.AcquireWorkspaceMemory(dev_ctx, "Out");
    pool_p->execute(astream,
                    {{DNNL_ARG_SRC, *src_memory},
                     {DNNL_ARG_DST, *dst_memory},
                     {DNNL_ARG_WORKSPACE, *workspace_memory}});
  } else {
    // Inference
    pool_p->execute(astream,
                    {{DNNL_ARG_SRC, *src_memory}, {DNNL_ARG_DST, *dst_memory}});
  }
  astream.wait();

  out->set_mem_desc(dst_memory->get_desc());
}
}  // namespace phi

PD_REGISTER_KERNEL(pool2d,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::Pool2dKernel,
                   float,
                   int8_t,
                   uint8_t,
                   phi::dtype::bfloat16) {}
