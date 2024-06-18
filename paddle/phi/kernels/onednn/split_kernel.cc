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

bool SplitCheckIfOneDNNSupport(const KernelContext* ctx) {
  if (ctx->InputAt<phi::DenseTensor>(0).mem_desc().get_inner_nblks() == 0) {
    return true;
  }
  return false;
}

const std::vector<int64_t> get_slice_strides(
    const std::vector<int64_t>& out_vec_dims,
    const dnnl::memory::desc& full_md,
    int axis) {
  auto strides = full_md.get_strides();
  auto ndims = full_md.get_dims().size();
  auto full_dims = full_md.get_dims();
  auto splitted_stride = strides[axis];
  std::vector<int64_t> slice_strides(ndims, splitted_stride);
  for (size_t i = 0; i < ndims; ++i) {
    slice_strides[i] = strides[i] > splitted_stride
                           ? (strides[i] / full_dims[axis]) * out_vec_dims[axis]
                           : strides[i];
  }
  return slice_strides;
}

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
  auto x_vec_dims = common::vectorize(x_dims);

  dnnl::memory::data_type x_type = funcs::ToOneDNNDataType(x.dtype());

  auto& astream = OneDNNContext::tls().get_stream();

  std::vector<int64_t> offset(x_vec_dims.size(), 0);
  funcs::ReorderOneDNNHandler reorder_handler(
      x_vec_dims, x.dtype(), x_type, onednn_engine);
  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      x.mem_desc(), funcs::to_void_cast(x.data<T>()));

  for (size_t i = 0; i < outs_number; ++i) {
    auto out_vec_dims = common::vectorize(out[i]->dims());
    auto slice_mem_p = reorder_handler.AcquireSubmemory(
        out_vec_dims, offset, reorder_src_memory_p);

    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
        out[i],
        out_vec_dims,
        get_slice_strides(out_vec_dims, x.mem_desc(), axis),
        dev_ctx.GetPlace());
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
  const std::vector<int64_t> sections_vec(num, input_axis_dim / num);

  IntArray sections(sections_vec);
  SplitKernel<T, Context>(dev_ctx, x, sections, axis_scalar, outs);
}

}  // namespace phi

PD_REGISTER_KERNEL(split,
                   OneDNN,
                   ONEDNN,
                   phi::SplitKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {
  kernel->check_if_onednn_kernel_support_ = phi::SplitCheckIfOneDNNSupport;
}

PD_REGISTER_KERNEL(split_with_num,
                   OneDNN,
                   ONEDNN,
                   phi::SplitWithNumKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {
  kernel->check_if_onednn_kernel_support_ = phi::SplitCheckIfOneDNNSupport;
}
