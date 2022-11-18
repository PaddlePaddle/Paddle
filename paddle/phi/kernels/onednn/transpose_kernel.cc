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

#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void TransposeKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int>& axis,
                     DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU,
      true,
      errors::PreconditionNotMet("Operator DNNL Transpose must use CPUPlace"));

  funcs::SetInMemDescWithLogicalLayoutFusesSupport(
      dev_ctx, const_cast<DenseTensor*>(&x), x.mem_desc());

  if (axis.size() == 1) {
    paddle::framework::TensorCopy(x, x.place(), out);
    out->set_mem_desc(x.mem_desc());
    return;
  }

  auto x_vec_dims = vectorize(x.dims());
  auto x_type = funcs::ToOneDNNDataType(x.dtype());
  const auto& onednn_engine = dev_ctx.GetEngine();
  funcs::ReorderOneDNNHandler reorder_handler(
      x_vec_dims, x.dtype(), x_type, onednn_engine);

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      x.mem_desc(), funcs::to_void_cast(x.data<T>()));

  auto dst_md =
      dnnl::memory::desc(x_vec_dims,
                         x.mem_desc().data_type(),
                         funcs::GetPlainOneDNNFormat(x_vec_dims.size()));
  // a trick is used here to fake transpose of out_md, so later it will be
  // "untransposed", leaving output data in plain format tag
  std::vector<int64_t> fake_strides(axis.size());
  auto dims = dst_md.dims();
  int total_stride = 1;
  for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
    fake_strides[axis[i]] = total_stride;
    total_stride *= dims[axis[i]];
  }

  dst_md =
      dnnl::memory::desc(x_vec_dims, x.mem_desc().data_type(), fake_strides);
  auto dst_data = dev_ctx.Alloc(out, x.type());

  auto reorder_dst_memory_p =
      std::make_shared<dnnl::memory>(dst_md, onednn_engine, dst_data);

  auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                  reorder_src_memory_p);

  auto& astream = OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
  astream.wait();

  // it is needed because oneDNN's permute axis understand axes order in
  // different way PaddlePaddle's transpose
  std::vector<int> permute_axis(axis.size());
  for (size_t i = 0; i < axis.size(); ++i) {
    permute_axis[axis[i]] = i;
  }
  funcs::SetOutMemDescWithLogicalLayoutFusesSupport(
      dev_ctx,
      out,
      reorder_dst_memory_p->get_desc().permute_axes(permute_axis));
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
