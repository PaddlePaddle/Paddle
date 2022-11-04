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

#include "paddle/phi/kernels/split_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const IntArray& sections,
                 const Scalar& split_axis,
                 std::vector<DenseTensor*> out) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  int axis = split_axis.to<int>();

  auto outs_number = out.size();
  const auto x_dims = x.dims();
  auto x_vec_dims = vectorize(x_dims);

  dnnl::memory::data_type x_type = funcs::ToOneDNNDataType(x.dtype());

  auto& astream = OneDNNContext::tls().get_stream();

  std::vector<int64_t> offset(x_vec_dims.size(), 0);
  funcs::ReorderOneDNNHandler reorder_handler(
      x_vec_dims, x.dtype(), x_type, onednn_engine);
  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      x.mem_desc(), funcs::to_void_cast(x.data<T>()));

  for (size_t i = 0; i < outs_number; ++i) {
    auto out_vec_dims = vectorize(out[i]->dims());
    auto slice_mem_p = reorder_handler.AcquireSubmemory(
        out_vec_dims, offset, reorder_src_memory_p);

    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
        out[i], out_vec_dims, x.format(), dev_ctx.GetPlace());
    auto reorder_p =
        reorder_handler.AcquireReorder(reorder_dst_memory_p, slice_mem_p);

    reorder_p->execute(astream, *slice_mem_p, *reorder_dst_memory_p);

    offset[axis] += sections.GetData()[i];
    out[i]->set_mem_desc(reorder_dst_memory_p->get_desc());
  }
  astream.wait();
}

template <typename T, typename Context>
void SplitWithNumKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        int num,
                        const Scalar& axis_scalar,
                        std::vector<DenseTensor*> outs) {
  int axis_value = axis_scalar.to<int>();
  auto input_axis_dim = x.dims().at(axis_value);
  std::vector<int64_t> sections_vec;
  for (int i = 0; i < num; ++i) {
    sections_vec.push_back(input_axis_dim / num);
  }
  IntArray sections(sections_vec);
  SplitKernel<T, Context>(dev_ctx, x, sections, axis_scalar, outs);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    split, OneDNN, ALL_LAYOUT, phi::SplitKernel, float, phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(split_with_num,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::SplitWithNumKernel,
                   float,
                   phi::dtype::bfloat16) {}
