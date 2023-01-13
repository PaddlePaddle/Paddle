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

#include "paddle/phi/kernels/transpose_grad_kernel.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void TransposeGradKernel(const Context& dev_ctx,
                         const DenseTensor& out_grad,
                         const std::vector<int>& axis,
                         DenseTensor* x_grad) {
  PADDLE_ENFORCE_EQ(dev_ctx.GetPlace().GetType() == AllocationType::CPU,
                    true,
                    errors::PreconditionNotMet(
                        "oneDNN TransposeGrad kernel must use CPUPlace"));
  if (!x_grad) return;

  const auto& onednn_engine = dev_ctx.GetEngine();

  if (axis.size() == 1) {
    Copy<Context>(dev_ctx, out_grad, out_grad.place(), false, x_grad);
    x_grad->set_mem_desc(out_grad.mem_desc());
    return;
  }

  std::vector<int64_t> out_grad_tz = vectorize(out_grad.dims());
  funcs::ReorderOneDNNHandler reorder_handler(
      out_grad_tz,
      out_grad.dtype(),
      funcs::ToOneDNNDataType(out_grad.dtype()),
      onednn_engine);

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      out_grad.mem_desc(), funcs::to_void_cast(out_grad.data<T>()));

  auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
      x_grad, out_grad.mem_desc(), dev_ctx.GetPlace());

  auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                  reorder_src_memory_p);

  auto& astream = OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
  astream.wait();
  x_grad->set_mem_desc(reorder_dst_memory_p->get_desc().permute_axes(axis));
}

}  // namespace phi

PD_REGISTER_KERNEL(
    transpose_grad, OneDNN, ONEDNN, phi::TransposeGradKernel, float) {}
