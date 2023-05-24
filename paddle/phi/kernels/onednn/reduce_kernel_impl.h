/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/phi/backends/onednn/onednn_reuse.h"

namespace phi {

inline std::vector<int64_t> CalculateReducedDims(
    const DenseTensor* input,
    const DenseTensor* output,
    const std::vector<int64_t>& reduce_dims,  // NOLINT
    bool reduce_all,
    bool keep_dim) {
  if (keep_dim) return vectorize(output->dims());

  if (reduce_all) return std::vector<int64_t>(input->dims().size(), 1);

  std::vector<int64_t> output_dims(vectorize(input->dims()));
  for (size_t i = 0; i < reduce_dims.size(); ++i) {
    // handle negative dims, f.e. "-1" means rightmost dimension
    int index = (reduce_dims[i] >= 0) ? reduce_dims[i]
                                      : input->dims().size() + reduce_dims[i];
    output_dims[index] = 1;
  }

  return output_dims;
}

template <typename T, typename Context>
void ReduceKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const IntArray& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DenseTensor* out,
                  dnnl::algorithm reduction_type) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  const auto& onednn_engine = dev_ctx.GetEngine();
  auto x_tz = vectorize(x.dims());
  auto out_tz =
      CalculateReducedDims(&x, out, dims.GetData(), reduce_all, keep_dim);

  auto& astream = OneDNNContext::tls().get_stream();

  // oneDNN reduce op does not support edge case in which memory is being
  // copied without actual reduction.
  // In that case reorder must be executed to maintain compatibility with
  // PaddlePaddle reduce op
  if (x_tz == out_tz) {
    dnnl::memory::data_type x_type = funcs::ToOneDNNDataType((x.dtype()));

    funcs::ReorderOneDNNHandler reorder_handler(
        x_tz, x.dtype(), x_type, onednn_engine);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        x.mem_desc(), funcs::to_void_cast(x.data<T>()));

    // reuse mem desc since it is a simple copy
    auto reorder_dst_memory_p =
        reorder_handler.AcquireDstMemory(out, x.mem_desc(), dev_ctx.GetPlace());

    auto reorder_p = reorder_handler.AcquireReorder(reorder_src_memory_p,
                                                    reorder_dst_memory_p);

    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
    astream.wait();

    const auto reshape_dims = out->dims().size() != 0
                                  ? vectorize<int64_t>(out->dims())
                                  : std::vector<int64_t>{1};
    out->set_mem_desc(reorder_dst_memory_p->get_desc().reshape(reshape_dims));
  } else {
    funcs::ReductionOneDNNHandler<T> handler(reduction_type,
                                             0.0f,
                                             0.0f,
                                             onednn_engine,
                                             dev_ctx.GetPlace(),
                                             &x,
                                             out,
                                             out_tz);

    auto src_memory_p = handler.AcquireSrcMemory(&x);
    auto dst_memory_p = handler.AcquireDstMemory(out);

    std::unordered_map<int, dnnl::memory> reduction_args = {
        {DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}};

    auto reduction_p = handler.AcquireForwardPrimitive();

    reduction_p->execute(astream, reduction_args);
    astream.wait();

    const auto reshape_dims = out->dims().size() != 0
                                  ? vectorize<int64_t>(out->dims())
                                  : std::vector<int64_t>{1};
    out->set_mem_desc(dst_memory_p->get_desc().reshape(reshape_dims));
  }
}

template <typename T, typename Context>
void ReduceGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      const IntArray& dims,
                      bool keep_dim,
                      bool reduce_all,
                      DenseTensor* x_grad,
                      dnnl::algorithm binary_type,
                      dnnl::algorithm reduction_type UNUSED,
                      float scale_x,
                      float scale_y) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  const auto& onednn_engine = dev_ctx.GetEngine();
  auto out_grad_tz = CalculateReducedDims(
      x_grad, &out_grad, dims.GetData(), reduce_all, keep_dim);
  auto x_grad_tz = vectorize(x_grad->dims());

  funcs::BroadcastDataOneDNNHandler<T> handler(binary_type,
                                               onednn_engine,
                                               dev_ctx.GetPlace(),
                                               &out_grad,
                                               x_grad,
                                               scale_x,
                                               scale_y,
                                               out_grad_tz);

  const auto src_memory_p = handler.AcquireSrcMemory(&out_grad);
  const auto dst_memory_p = handler.AcquireZeroedDstMemory(x_grad);
  const auto binary_prim = handler.AcquireForwardPrimitive();

  const std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC_0, *dst_memory_p},
      {DNNL_ARG_SRC_1, *src_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}};

  auto& astream = OneDNNContext::tls().get_stream();
  binary_prim->execute(astream, args);
  astream.wait();

  x_grad->set_mem_desc(dst_memory_p->get_desc());
}

}  // namespace phi
