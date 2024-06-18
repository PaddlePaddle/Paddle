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

#include "paddle/phi/kernels/slice_grad_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

bool SliceGradCheckIfOneDNNSupport(const KernelContext* ctx) {
  if (ctx->InputAt<phi::DenseTensor>(1).mem_desc().get_inner_nblks() == 0) {
    return true;
  }
  return false;
}

template <typename T, typename Context>
void SliceGradKernel(const Context& dev_ctx,
                     const DenseTensor& input UNUSED,
                     const DenseTensor& out_grad,
                     const std::vector<int64_t>& axes,
                     const IntArray& starts,
                     const IntArray& ends,
                     const std::vector<int64_t>& infer_flags UNUSED,
                     const std::vector<int64_t>& decrease_axis UNUSED,
                     DenseTensor* input_grad) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  auto dx_dims = common::vectorize(input_grad->dims());

  auto starts_vec = starts.GetData();
  auto ends_vec = ends.GetData();

  std::vector<int64_t> offsets(dx_dims.size(), 0);
  std::vector<int64_t> slice_dims(dx_dims);

  for (size_t i = 0; i < axes.size(); ++i) {
    starts_vec[i] =
        starts_vec[i] < 0 ? dx_dims[axes[i]] + starts_vec[i] : starts_vec[i];
    ends_vec[i] = ends_vec[i] < 0 ? dx_dims[axes[i]] + ends_vec[i]
                                  : std::min(ends_vec[i], dx_dims[axes[i]]);
    offsets[axes[i]] = starts_vec[i];
    slice_dims[axes[i]] = ends_vec[i] - starts_vec[i];
  }

  dnnl::memory::data_type out_grad_type =
      funcs::ToOneDNNDataType(out_grad.dtype());

  funcs::ReorderOneDNNHandler reorder_handler(
      slice_dims, out_grad.dtype(), out_grad_type, onednn_engine);

  auto reorder_src_memory_p =
      reorder_handler.AcquireSrcMemory(out_grad.mem_desc().reshape(slice_dims),
                                       funcs::to_void_cast(out_grad.data<T>()));
  auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
      input_grad,
      dx_dims,
      funcs::GetPlainOneDNNFormat(static_cast<int>(dx_dims.size())),
      dev_ctx.GetPlace());
  memset(input_grad->data<T>(), 0, reorder_dst_memory_p->get_desc().get_size());

  auto slice_mem_p = reorder_handler.AcquireSubmemory(
      slice_dims, offsets, reorder_dst_memory_p);

  auto reorder_p =
      reorder_handler.AcquireReorder(slice_mem_p, reorder_src_memory_p);
  auto& astream = OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, *reorder_src_memory_p, *slice_mem_p);
  astream.wait();

  input_grad->set_mem_desc(reorder_dst_memory_p->get_desc());
}

}  // namespace phi

PD_REGISTER_KERNEL(slice_grad,
                   OneDNN,
                   ONEDNN,
                   phi::SliceGradKernel,
                   float,
                   phi::dtype::bfloat16) {
  kernel->check_if_onednn_kernel_support_ = phi::SliceGradCheckIfOneDNNSupport;
}
