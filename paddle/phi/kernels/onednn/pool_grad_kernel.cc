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
template <typename T, typename Context>
void Pool2dGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& out,
                      const DenseTensor& dout,
                      const std::vector<int>& kernel_size,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      bool ceil_mode,
                      bool exclusive,
                      const std::string& data_format,
                      const std::string& pooling_type,
                      bool global_pooling,
                      bool adaptive,
                      const std::string& padding_algorithm,
                      DenseTensor* dx) {
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
                                         &dout,
                                         dx);

  auto diff_dst_memory = handler.AcquireDiffDstMemory(&dout);
  auto diff_src_memory = handler.AcquireDiffSrcMemory(dx);

  auto pool_bwd_p = handler.AcquireBackwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  if (pooling_type == "max") {
    // Max - pooling needs Workspace
    auto workspace_memory = handler.AcquireWorkspaceMemory(dev_ctx, "Out");
    pool_bwd_p->execute(astream,
                        {{DNNL_ARG_DIFF_SRC, *diff_src_memory},
                         {DNNL_ARG_DIFF_DST, *diff_dst_memory},
                         {DNNL_ARG_WORKSPACE, *workspace_memory}});
  } else {
    // Average Pooling
    pool_bwd_p->execute(astream,
                        {{DNNL_ARG_DIFF_SRC, *diff_src_memory},
                         {DNNL_ARG_DIFF_DST, *diff_dst_memory}});
  }
  astream.wait();

  dx->set_mem_desc(diff_src_memory->get_desc());
}
}  // namespace phi

PD_REGISTER_KERNEL(pool2d_grad,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::Pool2dGradKernel,
                   float,
                   phi::dtype::bfloat16) {}
