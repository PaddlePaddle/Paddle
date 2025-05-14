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
  bool has_zero_size = false;

  for (size_t i = 0; i < out_new_dims.size(); ++i) {
    out_new_dims[i] = out_new_dims[i] >= 0 ? out_new_dims[i] : x_vec_dims[i];
  }

  if (x_vec_dims.size() != out_new_dims.size()) {
    x_vec_dims = GetExtendedXDims(x_vec_dims, out_new_dims.size());  // NOLINT
  }

  for (size_t i = 0; i < x_vec_dims.size(); ++i) {
    PADDLE_ENFORCE_GE(
        out_new_dims[i],
        0,
        common::errors::InvalidArgument(
            "The expanded size (%d) for non-existing dimensions must be "
            "positive for expand_v2 op.",
            out_new_dims[i]));

    PADDLE_ENFORCE_GE(
        x_vec_dims[i],
        0,
        common::errors::InvalidArgument(
            "The expanded size (%d) for non-existing dimensions must be "
            "positive for expand_v2 op.",
            x_vec_dims[i]));

    PADDLE_ENFORCE_EQ(
        x_vec_dims[i] == 1 || x_vec_dims[i] == out_new_dims[i],
        true,
        common::errors::InvalidArgument(
            "The value (%d) of the non-singleton dimension does not match"
            " the corresponding value (%d) in shape for expand_v2 op.",
            x_vec_dims[i],
            out_new_dims[i]));
    if (out_new_dims[i] == 0) {
      has_zero_size = true;
    }
  }

  out->Resize(common::make_ddim(out_new_dims));
  if (has_zero_size) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
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
