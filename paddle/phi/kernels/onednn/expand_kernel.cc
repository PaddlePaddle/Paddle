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

#include "paddle/phi/kernels/expand_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

std::vector<int64_t> GetExtendedXDims(const std::vector<int64_t>& x_vec_dims,
                                      int new_size) {
  std::vector<int64_t> extended_x_dims(new_size, 1);
  std::copy(x_vec_dims.begin(),
            x_vec_dims.end(),
            extended_x_dims.begin() + new_size - x_vec_dims.size());  // NOLINT

  return extended_x_dims;
}

template <typename T, typename Context>
void ExpandKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const IntArray& shape,
                  DenseTensor* out) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  auto x_vec_dims = common::vectorize(x.dims());

  auto out_new_dims = shape.GetData();

  for (size_t i = 0; i < out_new_dims.size(); ++i) {
    out_new_dims[i] = out_new_dims[i] > 0 ? out_new_dims[i] : x_vec_dims[i];
  }

  if (x_vec_dims.size() != out_new_dims.size()) {
    x_vec_dims = GetExtendedXDims(x_vec_dims, out_new_dims.size());  // NOLINT
  }

  out->Resize(common::make_ddim(out_new_dims));
  funcs::BroadcastDataOneDNNHandler<T> handler(dnnl::algorithm::binary_add,
                                               onednn_engine,
                                               dev_ctx.GetPlace(),
                                               &x,
                                               out,
                                               0.0f,
                                               1.0f,
                                               x_vec_dims);

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  auto dst_memory_p = handler.AcquireZeroedDstMemory(out);
  auto binary_p = handler.AcquireForwardPrimitive();

  const std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC_0, *dst_memory_p},
      {DNNL_ARG_SRC_1, *src_memory_p},
      {DNNL_ARG_DST, *dst_memory_p},
      {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, handler.Get_Scale_Memory(0.0f)},
      {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, handler.Get_Scale_Memory(1.0f)}};

  auto& astream = OneDNNContext::tls().get_stream();
  binary_p->execute(astream, args);
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}
}  // namespace phi

PD_REGISTER_KERNEL(
    expand, OneDNN, ONEDNN, phi::ExpandKernel, float, phi::dtype::bfloat16) {}
