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
<<<<<<< HEAD
=======

#include "paddle/fluid/framework/tensor_util.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void TransposeGradKernel(const Context& dev_ctx,
                         const DenseTensor& out_grad,
                         const std::vector<int>& axis,
                         DenseTensor* x_grad) {
<<<<<<< HEAD
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
=======
  PADDLE_ENFORCE_EQ(dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU,
                    true,
                    errors::PreconditionNotMet(
                        "Operator DNNL TransposeGrad must use CPUPlace"));
  if (!x_grad) return;

  const auto& onednn_engine = dev_ctx.GetEngine();
  std::vector<int> reversed_axis(axis);
  if (axis.size() == 1) {
    paddle::framework::TensorCopy(out_grad, out_grad.place(), x_grad);
    x_grad->set_format(out_grad.format());
    return;
  }

  for (size_t i = 0; i < axis.size(); i++) {
    reversed_axis[axis[i]] = i;
  }

  const T* out_grad_data = out_grad.data<T>();
  dev_ctx.template Alloc<T>(x_grad);
  auto nchw_tz = vectorize<int64_t>(out_grad.dims());

  funcs::TransposeOneDNNHandler<T> handler(
      dev_ctx, nchw_tz, reversed_axis, onednn_engine);

  auto transpose_src_memory_p = handler.AcquireSrcMemory(
      out_grad.format(), funcs::to_void_cast<T>(out_grad_data));
  auto transpose_dst_memory_p =
      handler.AcquireDstMemory(x_grad, dev_ctx.GetPlace());
  auto transpose_p =
      handler.AcquireTranspose(transpose_dst_memory_p, transpose_src_memory_p);

  auto& astream = OneDNNContext::tls().get_stream();
  transpose_p->execute(
      astream, *transpose_src_memory_p, *transpose_dst_memory_p);
  astream.wait();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

}  // namespace phi

PD_REGISTER_KERNEL(
<<<<<<< HEAD
    transpose_grad, OneDNN, ONEDNN, phi::TransposeGradKernel, float) {}
=======
    transpose_grad, OneDNN, ALL_LAYOUT, phi::TransposeGradKernel, float) {}
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
