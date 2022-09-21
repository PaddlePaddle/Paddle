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

#include "paddle/phi/kernels/squeeze_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

template <typename T, typename Context>
void SqueezeKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const IntArray& axes,
                   DenseTensor* out) {
  phi::DDim x_dims = x.dims();
  phi::DDim out_dims = GetOutputShape(axes, x_dims, true);

  auto x_vec_dims = phi::vectorize(x_dims);
  phi::funcs::ReorderOneDNNHandler reorder_handler(
      x_vec_dims,
      x.dtype(),
      phi::funcs::ToOneDNNDataType(x.dtype()),
      dev_ctx.GetEngine());

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      x.mem_desc(), const_cast<void*>(x.data()));
  out->Resize(x_dims);  // to match x numel, format is changed later
  // reorder is done into a plain tag to allow usage with blocked formats
  auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
      out, getPlainFormatTag(x), dev_ctx.GetPlace());
  auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                  reorder_src_memory_p);

  auto& astream = phi::OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);

  astream.wait();

  out->Resize(out_dims);
  out->set_mem_desc(
      reorder_dst_memory_p->get_desc().reshape(phi::vectorize(out_dims)));
}

template <typename T, typename Context>
void SqueezeWithXShapeKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const IntArray& axes,
                             DenseTensor* out,
                             DenseTensor* xshape) {
  SqueezeKernel<T, Context>(dev_ctx, x, axes, out);
}

PD_REGISTER_KERNEL(squeeze,
                   CPU,
                   ALL_LAYOUT,
                   phi::SqueezeKernel,
                   float,
                   phi::dtype::bfloat16, ) {}

PD_REGISTER_KERNEL(squeeze_with_xshape,
                   CPU,
                   ALL_LAYOUT,
                   phi::SqueezeWithXShapeKernel,
                   float,
                   phi::dtype::bfloat16) {}
