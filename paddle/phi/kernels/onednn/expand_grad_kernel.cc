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

#include "paddle/phi/kernels/expand_grad_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void ExpandGradKernel(const Context& dev_ctx,
                      const DenseTensor& x UNUSED,
                      const DenseTensor& out_grad,
                      const IntArray& shape UNUSED,
                      DenseTensor* in_grad) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  auto in_grad_vec_dims = common::vectorize(in_grad->dims());
  auto out_grad_vec_dims = common::vectorize(out_grad.dims());

  if (in_grad_vec_dims.size() != out_grad_vec_dims.size()) {
    in_grad_vec_dims.insert(in_grad_vec_dims.begin(),
                            out_grad_vec_dims.size() - in_grad_vec_dims.size(),
                            1);
  }

  auto& astream = OneDNNContext::tls().get_stream();
  if (out_grad_vec_dims == in_grad_vec_dims) {
    dnnl::memory::data_type out_grad_type =
        funcs::ToOneDNNDataType(out_grad.dtype());
    if (out_grad_vec_dims.empty()) {
      out_grad_vec_dims = std::vector<int64_t>{1};
    }
    funcs::ReorderOneDNNHandler reorder_handler(
        out_grad_vec_dims, out_grad.dtype(), out_grad_type, onednn_engine);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        out_grad.mem_desc(), funcs::to_void_cast(out_grad.data<T>()));

    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
        in_grad,
        funcs::GetPlainOneDNNFormat(static_cast<int>(in_grad_vec_dims.size())),
        dev_ctx.GetPlace());

    auto reorder_p = reorder_handler.AcquireReorder(reorder_src_memory_p,
                                                    reorder_dst_memory_p);

    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
    astream.wait();

    in_grad->set_mem_desc(reorder_dst_memory_p->get_desc());
  } else {
    funcs::ReductionOneDNNHandler<T> handler(dnnl::algorithm::reduction_sum,
                                             0.0f,
                                             0.0f,
                                             onednn_engine,
                                             dev_ctx.GetPlace(),
                                             &out_grad,
                                             in_grad,
                                             in_grad_vec_dims);

    auto src_memory_p = handler.AcquireSrcMemory(&out_grad);
    auto dst_memory_p = handler.AcquireDstMemory(in_grad);

    std::unordered_map<int, dnnl::memory> reduction_args = {
        {DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}};

    auto reduction_p = handler.AcquireForwardPrimitive();

    reduction_p->execute(astream, reduction_args);
    astream.wait();
    in_grad->set_layout(DataLayout::ONEDNN);
    const auto in_grad_md_dims =
        in_grad->dims().size() != 0
            ? common::vectorize<int64_t>(in_grad->dims())
            : std::vector<int64_t>{1};
    in_grad->set_mem_desc(dst_memory_p->get_desc().reshape(in_grad_md_dims));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(expand_grad,
                   OneDNN,
                   ONEDNN,
                   phi::ExpandGradKernel,
                   float,
                   phi::dtype::bfloat16) {}
