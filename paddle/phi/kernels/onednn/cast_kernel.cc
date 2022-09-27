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

#include "paddle/phi/kernels/cast_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DataType out_dtype,
                DenseTensor* out) {
  DataType in_dtype = x.dtype();

  dnnl::memory::data_type in_dnnl_dtype = funcs::ToOneDNNDataType(in_dtype);
  dnnl::memory::data_type out_dnnl_dtype = funcs::ToOneDNNDataType(out_dtype);

  auto x_tz = phi::vectorize(x.dims());

  funcs::ReorderOneDNNHandler reorder_handler(x_tz,
                                              in_dtype,
                                              in_dnnl_dtype,
                                              out_dtype,
                                              out_dnnl_dtype,
                                              dev_ctx.GetEngine());

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      x.mem_desc(), funcs::to_void_cast(x.data<T>()));
  auto reorder_dst_memory_p =
      reorder_handler.AcquireDstMemory(out, x.mem_desc(), dev_ctx.GetPlace());
  auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                  reorder_src_memory_p);

  auto& astream = OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
  astream.wait();

  out->set_layout(DataLayout::ONEDNN);
  out->set_mem_desc(reorder_dst_memory_p->get_desc());
}

}  // namespace phi

PD_REGISTER_KERNEL(
    cast, OneDNN, ALL_LAYOUT, phi::CastKernel, float, phi::dtype::bfloat16) {}
