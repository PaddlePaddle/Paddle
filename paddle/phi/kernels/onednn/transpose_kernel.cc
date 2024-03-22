// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/transpose_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void TransposeKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int>& axis,
                     DenseTensor* out) {
  // Here we need to match dims to paddle layout
  // as we are producing non-oneDNN result
  auto x_dims = x.dims();
  if ((x_dims.size() >= 3) &&
      (phi::OneDNNContext::tls().get_cur_paddle_data_layout() ==
       phi::DataLayout::kNHWC)) {
    int axis_size = static_cast<int>(axis.size());
    std::vector<int> formatted_axis = axis;
    std::vector<int> count(axis_size, 0);
    for (int i = 0; i < axis_size; i++) {
      if (axis[i] < 0) {
        formatted_axis[i] = axis[i] + axis_size;
      }
    }
    auto dims = common::vectorize<int>(x_dims);

    std::rotate(dims.begin() + 1, dims.begin() + 2, dims.end());
    x_dims = x_dims.reshape(dims);
    VLOG(3)
        << "Rotating Shape in Transpose from: kMKLDNN to: kNHWC output_shape";

    phi::DDim out_dims(x_dims);
    for (size_t i = 0; i < axis.size(); i++) {
      out_dims[i] = x_dims[formatted_axis[i]];  // NOLINT
    }
    out->Resize(out_dims);
  }

  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType(),
      AllocationType::CPU,
      errors::PreconditionNotMet("oneDNN Transpose kernel must use CPUPlace"));

  if (axis.size() == 1 || axis.empty()) {
    Copy<Context>(dev_ctx, x, x.place(), false, out);
    out->set_mem_desc(x.mem_desc());
    return;
  }

  auto x_vec_dims = common::vectorize(x.dims());
  auto x_type = funcs::ToOneDNNDataType(x.dtype());
  funcs::ReorderOneDNNHandler reorder_handler(
      x_vec_dims, x.dtype(), x_type, dev_ctx.GetEngine());
  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      x.mem_desc(), funcs::to_void_cast(x.data<T>()));

  auto fake_strides = funcs::FakeTransposeStrides(x_vec_dims, axis);
  auto dst_md = dnnl::memory::desc(
      x_vec_dims, x.mem_desc().get_data_type(), fake_strides);
  auto reorder_dst_memory_p =
      reorder_handler.AcquireDstMemory(out, dst_md, dev_ctx.GetPlace());
  auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                  reorder_src_memory_p);

  auto& astream = OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
  astream.wait();
  out->set_mem_desc(reorder_dst_memory_p->get_desc().permute_axes(
      funcs::TransposeToPermuteAxes(axis)));
}
}  // namespace phi

PD_REGISTER_KERNEL(transpose,
                   OneDNN,
                   ONEDNN,
                   phi::TransposeKernel,
                   float,
                   uint8_t,
                   int8_t,
                   phi::dtype::bfloat16) {}
