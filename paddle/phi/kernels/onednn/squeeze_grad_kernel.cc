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

#include "paddle/phi/kernels/squeeze_grad_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SqueezeGradKernel(const Context& dev_ctx,
                       const DenseTensor& xshape,
                       const DenseTensor& dout,
                       const IntArray& axes UNUSED,
                       DenseTensor* dx) {
  auto dout_vec_dims = dout.dims().size() != 0 ? common::vectorize(dout.dims())
                                               : std::vector<int64_t>{1};

  auto dout_type = funcs::ToOneDNNDataType(dout.dtype());

  funcs::ReorderOneDNNHandler reorder_handler(
      dout_vec_dims, dout.dtype(), dout_type, dev_ctx.GetEngine());

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      dout.mem_desc(), funcs::to_void_cast(dout.data<T>()));
  auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
      dx,
      funcs::GetPlainOneDNNFormat(static_cast<int>(dout_vec_dims.size())),
      dev_ctx.GetPlace());
  auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                  reorder_src_memory_p);

  auto& astream = OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
  astream.wait();

  auto dx_dims = slice_ddim(xshape.dims(), 1, xshape.dims().size());
  dx->Resize(dx_dims);
  reorder_dst_memory_p->get_desc().reshape(common::vectorize(dx_dims));
}

}  // namespace phi

PD_REGISTER_KERNEL(squeeze_grad,
                   OneDNN,
                   ONEDNN,
                   phi::SqueezeGradKernel,
                   float,
                   phi::dtype::bfloat16) {}
