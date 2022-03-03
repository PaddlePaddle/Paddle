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

#include "paddle/phi/kernels/graph_send_recv_grad_kernel.h"
#include "paddle/phi/kernels/cpu/graph_send_recv_funcs.h"

#include <algorithm>
#include <vector>

#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename IndexT, typename Functor>
void graph_send_recv_cpu_for_loop_grad(const int& input_size,
                                       const int& index_size,
                                       const IndexT* s_index,
                                       const IndexT* d_index,
                                       const DenseTensor& src,
                                       DenseTensor* dst,
                                       const std::string& pool_type,
                                       const int* dst_count = nullptr,
                                       const DenseTensor* input = nullptr,
                                       const DenseTensor* output = nullptr) {
  if (pool_type == "SUM") {
    Functor functor;
    for (int i = 0; i < index_size; ++i) {
      const IndexT& src_idx = s_index[i];
      const IndexT& dst_idx = d_index[i];
      elementwise_inner_operation<T, IndexT, Functor>(
          src, dst, src_idx, dst_idx, false, functor);
    }
  } else if (pool_type == "MEAN") {
    for (int i = 0; i < index_size; ++i) {
      const IndexT& src_idx = s_index[i];
      const IndexT& dst_idx = d_index[i];
      auto src_slice = src.Slice(src_idx, src_idx + 1);
      auto dst_slice = dst->Slice(dst_idx, dst_idx + 1);
      auto eigen_src = phi::EigenVector<T>::Flatten(src_slice);
      auto eigen_dst = phi::EigenVector<T>::Flatten(dst_slice);
      eigen_dst += (eigen_src / static_cast<T>(dst_count[src_idx]));
    }
  } else if (pool_type == "MIN" || pool_type == "MAX") {
    for (int i = 0; i < index_size; ++i) {
      const IndexT& forward_src_idx = d_index[i];
      const IndexT& forward_dst_idx = s_index[i];
      auto input_slice = input->Slice(forward_src_idx, forward_src_idx + 1);
      auto output_slice = output->Slice(forward_dst_idx, forward_dst_idx + 1);
      auto eigen_input = phi::EigenVector<T>::Flatten(input_slice);
      auto eigen_output = phi::EigenVector<T>::Flatten(output_slice);

      auto src_slice = src.Slice(forward_dst_idx, forward_dst_idx + 1);
      auto dst_slice = dst->Slice(forward_src_idx, forward_src_idx + 1);
      auto eigen_src = phi::EigenVector<T>::Flatten(src_slice);
      auto eigen_dst = phi::EigenVector<T>::Flatten(dst_slice);
      eigen_dst += eigen_src * (eigen_output == eigen_input);
    }
  }
}

template <typename Context, typename T, typename IndexT>
void GraphSendRecvGradOpKernelLaunchHelper(
    const Context& ctx,
    const DenseTensor& out_grad,
    const DenseTensor& src_index,
    const DenseTensor& dst_index,
    const std::string& pool_type,
    DenseTensor* x_grad,
    const DenseTensor* dst_count = nullptr,
    const DenseTensor* x = nullptr,
    const DenseTensor* out = nullptr) {
  const int& index_size = dst_index.dims()[0];

  ctx.template Alloc<T>(x_grad);
  T* p_output = x_grad->data<T>();
  const auto& src_dims = out_grad.dims();
  int64_t memset_size = 1;
  for (int i = 0; i < src_dims.size(); ++i) memset_size *= src_dims[i];
  const size_t& memset_bytes = memset_size * sizeof(T);
  memset(p_output, 0, memset_bytes);

  if (index_size == 0) return;

  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();

  if (pool_type == "SUM") {
    graph_send_recv_cpu_for_loop_grad<T, IndexT, GraphSendRecvSumFunctor<T>>(
        src_dims[0], index_size, d_index, s_index, out_grad, x_grad, pool_type);
  } else if (pool_type == "MEAN") {
    const int* s_count = dst_count->data<int>();
    // Functor not used here.
    graph_send_recv_cpu_for_loop_grad<T, IndexT, GraphSendRecvSumFunctor<T>>(
        src_dims[0],
        index_size,
        d_index,
        s_index,
        out_grad,
        x_grad,
        pool_type,
        s_count);
  } else if (pool_type == "MIN" || pool_type == "MAX") {
    // Functor not used here.
    graph_send_recv_cpu_for_loop_grad<T, IndexT, GraphSendRecvMinFunctor<T>>(
        src_dims[0],
        index_size,
        d_index,
        s_index,
        out_grad,
        x_grad,
        pool_type,
        nullptr,
        x,
        out);
  }
}

template <typename T, typename Context>
void GraphSendRecvGradKernel(const Context& ctx,
                             const DenseTensor& out_grad,
                             paddle::optional<const DenseTensor&> x,
                             paddle::optional<const DenseTensor&> out,
                             const DenseTensor& src_index,
                             const DenseTensor& dst_index,
                             paddle::optional<const DenseTensor&> dst_count,
                             const std::string& pool_type,
                             DenseTensor* x_grad) {
  auto index_type = src_index.dtype();
  if (index_type == phi::DataType::INT32) {
    GraphSendRecvGradOpKernelLaunchHelper<Context, T, int32_t>(
        ctx,
        out_grad,
        src_index,
        dst_index,
        pool_type,
        x_grad,
        dst_count.get_ptr(),
        x.get_ptr(),
        out.get_ptr());
  } else if (index_type == phi::DataType::INT64) {
    GraphSendRecvGradOpKernelLaunchHelper<Context, T, int64_t>(
        ctx,
        out_grad,
        src_index,
        dst_index,
        pool_type,
        x_grad,
        dst_count.get_ptr(),
        x.get_ptr(),
        out.get_ptr());
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(graph_send_recv_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::GraphSendRecvGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
