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

#include "paddle/phi/kernels/slice_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SliceRawKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<int64_t>& axes,
                    const IntArray& starts,
                    const IntArray& ends,
                    const std::vector<int64_t>& infer_flags,
                    const std::vector<int64_t>& decrease_axis,
                    DenseTensor* out) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  auto x_vec_dims = vectorize(x.dims());

  auto starts_vec = starts.GetData();
  auto ends_vec = ends.GetData();

  std::vector<int64_t> offsets(x_vec_dims.size(), 0);
  std::vector<int64_t> slice_dims(x_vec_dims);

  for (size_t i = 0; i < axes.size(); ++i) {
    starts_vec[i] =
        starts_vec[i] < 0 ? x_vec_dims[axes[i]] + starts_vec[i] : starts_vec[i];
    ends_vec[i] = ends_vec[i] < 0 ? x_vec_dims[axes[i]] + ends_vec[i]
                                  : std::min(ends_vec[i], x_vec_dims[axes[i]]);
    offsets[axes[i]] = starts_vec[i];
    slice_dims[axes[i]] =
        std::max(static_cast<int64_t>(0), ends_vec[i] - starts_vec[i]);
  }

  out->Resize(make_ddim(slice_dims));

  // Note(0x45f): To support slice Tensors with shapes like [0, 0, 0].
  if (!x.initialized()) {
    dev_ctx.Alloc(out, x.dtype());
    out->set_layout(DataLayout::ONEDNN);
    return;
  }

  dnnl::memory::data_type x_type = funcs::ToOneDNNDataType(x.dtype());

  funcs::ReorderOneDNNHandler reorder_handler(
      x_vec_dims, x.dtype(), x_type, onednn_engine);

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      x.mem_desc(), funcs::to_void_cast(x.data<T>()));
  auto slice_mem_p = reorder_handler.AcquireSubmemory(
      slice_dims, offsets, reorder_src_memory_p);
  auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
      out,
      slice_dims,
      funcs::GetPlainOneDNNFormat(x_vec_dims.size()),
      dev_ctx.GetPlace());

  auto reorder_p =
      reorder_handler.AcquireReorder(reorder_dst_memory_p, slice_mem_p);
  auto& astream = OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, *slice_mem_p, *reorder_dst_memory_p);

  std::vector<int64_t> new_out_dims(slice_dims.size() - decrease_axis.size());

  if (new_out_dims.size() == 0) {
    new_out_dims.emplace_back(1);
  } else {
    for (const auto& axis : decrease_axis) {
      slice_dims[axis] = 0;
    }

    int i = 0;
    for (const auto& slice_dim : slice_dims) {
      if (slice_dim != 0) new_out_dims[i++] = slice_dim;
    }
  }

  astream.wait();
  out->Resize(make_ddim(new_out_dims));
  out->set_mem_desc(reorder_dst_memory_p->get_desc().reshape(new_out_dims));
}

}  // namespace phi

PD_REGISTER_KERNEL(slice,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::SliceRawKernel,
                   float,
                   int8_t,
                   uint8_t,
                   phi::dtype::bfloat16) {}
