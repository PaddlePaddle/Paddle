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

#include "paddle/phi/kernels/concat_grad_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"

namespace phi {

template <typename T, typename Context>
void ConcatGradKernel(const Context& dev_ctx,
                      const std::vector<const DenseTensor*>& x,
                      const DenseTensor& out_grad,
                      const Scalar& axis_scalar,
                      std::vector<DenseTensor*> x_grad) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  auto& astream = OneDNNContext::tls().get_stream();

  for (size_t i = 0; i < x_grad.size(); ++i) {
    if (x_grad[i] != nullptr) {
      x_grad[i]->set_lod(x[i]->lod());
    }
  }

  int axis = axis_scalar.to<int>();

  auto out_grad_vec_dims = common::vectorize(out_grad.dims());

  axis = static_cast<int>(funcs::ComputeAxis(axis, out_grad_vec_dims.size()));

  std::vector<int64_t> offset(out_grad_vec_dims.size(), 0);

  dnnl::memory::data_type out_grad_type =
      funcs::ToOneDNNDataType(out_grad.dtype());
  funcs::ReorderOneDNNHandler reorder_handler(
      out_grad_vec_dims, out_grad.dtype(), out_grad_type, onednn_engine);
  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      out_grad.mem_desc(), funcs::to_void_cast(out_grad.data<T>()));

  for (auto& grad : x_grad) {
    if (grad && grad->numel() != 0UL) {
      auto x_grad_vec_dims = common::vectorize(grad->dims());
      auto slice_mem_p = reorder_handler.AcquireSubmemory(
          x_grad_vec_dims, offset, reorder_src_memory_p);

      auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
          grad,
          x_grad_vec_dims,
          funcs::GetPlainOneDNNFormat(static_cast<int>(x_grad_vec_dims.size())),
          dev_ctx.GetPlace());
      auto reorder_p =
          reorder_handler.AcquireReorder(reorder_dst_memory_p, slice_mem_p);

      reorder_p->execute(astream, *slice_mem_p, *reorder_dst_memory_p);

      offset[axis] += grad->dims()[axis];

      grad->set_mem_desc(reorder_dst_memory_p->get_desc());
    }
  }
  astream.wait();
}
}  // namespace phi

PD_REGISTER_KERNEL(concat_grad,
                   OneDNN,
                   ONEDNN,
                   phi::ConcatGradKernel,
                   float,
                   phi::dtype::bfloat16) {}
