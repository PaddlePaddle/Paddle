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
#include "paddle/phi/kernels/funcs/unsqueeze.h"

namespace phi {

template <typename T, typename Context>
void ExecuteSqueeze(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DDim& x_dims,
                    const DDim& out_dims,
                    DenseTensor* out) {
  auto x_vec_dims = common::vectorize(x_dims);

  funcs::ReorderOneDNNHandler reorder_handler(
      x_vec_dims,
      x.dtype(),
      funcs::ToOneDNNDataType(x.dtype()),
      dev_ctx.GetEngine());

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      x.mem_desc(), funcs::to_void_cast(x.data<T>()));
  out->Resize(x_dims);  // to match x numel, format is changed later
  // reorder is done into a plain tag to allow usage with blocked formats
  auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
      out, funcs::GetPlainOneDNNFormat(x_dims.size()), dev_ctx.GetPlace());
  auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                  reorder_src_memory_p);
  auto& astream = OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
  astream.wait();

  out->Resize(out_dims);

  auto reshape_dims = out_dims.size() != 0 ? common::vectorize(out_dims)
                                           : std::vector<int64_t>{1};
  out->set_mem_desc(reorder_dst_memory_p->get_desc().reshape(reshape_dims));
}

template <typename T, typename Context>
void SqueezeKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const IntArray& axes,
                   DenseTensor* out) {
  auto x_dims = x.dims();
  auto x_dims_tz = x_dims.size();
  std::vector<int32_t> tmp(axes.GetData().begin(), axes.GetData().end());

  // Currently there is only transformation for tensors, while attr axes still
  // follows default dtype instead of oneDNN dtype, so here manually change it
  if ((x_dims_tz >= 3) &&
      (phi::OneDNNContext::tls().get_cur_paddle_data_layout() ==
           phi::DataLayout::NDHWC ||
       phi::OneDNNContext::tls().get_cur_paddle_data_layout() ==
           phi::DataLayout::NHWC)) {
    int axes_size = tmp.size();
    for (int i = 0; i < axes_size; i++) {
      if (tmp[i] < 0) {
        tmp[i] += x_dims_tz;
      }
      if (tmp[i] >= 1 && tmp[i] < (x_dims_tz - 1)) {
        tmp[i] += 1;
      } else if (tmp[i] == (x_dims_tz - 1)) {
        tmp[i] = 1;
      }
    }
  }

  auto out_dims = funcs::GetOutputSqueezeShape(tmp, x_dims, true);
  ExecuteSqueeze<T, Context>(dev_ctx, x, x_dims, out_dims, out);
}

template <typename T, typename Context>
void SqueezeWithXShapeKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const IntArray& axes,
                             DenseTensor* out,
                             DenseTensor* xshape) {
  if (xshape == nullptr) {
    SqueezeKernel<T, Context>(dev_ctx, x, axes, out);
  } else {
    auto x_dims = slice_ddim(xshape->dims(), 1, xshape->dims().size());
    auto out_dims = out->dims();
    ExecuteSqueeze<T, Context>(dev_ctx, x, x_dims, out_dims, out);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    squeeze, OneDNN, ONEDNN, phi::SqueezeKernel, float, phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(squeeze_with_xshape,
                   OneDNN,
                   ONEDNN,
                   phi::SqueezeWithXShapeKernel,
                   float,
                   phi::dtype::bfloat16) {}
