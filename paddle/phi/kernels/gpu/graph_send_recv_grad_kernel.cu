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

#include "paddle/phi/kernels/gpu/graph_send_recv_funcs.h"
#include "paddle/phi/kernels/graph_send_recv_grad_kernel.h"

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename Context, typename T, typename IndexT>
void GraphSendRecvGradOpCUDAKernelLaunchHelper(
    const Context& ctx,
    const DenseTensor& out_grad,
    const DenseTensor& x,
    const DenseTensor& src_index,
    const DenseTensor& dst_index,
    const std::string& pool_type,
    DenseTensor* x_grad,
    const DenseTensor* dst_count = nullptr,
    const DenseTensor* out = nullptr) {
  const int& index_size = dst_index.dims()[0];

  ctx.template Alloc<T>(x_grad);
  T* p_output = x_grad->data<T>();

  const auto& src_dims = x.dims();
  int64_t memset_size = 1;
  for (int i = 0; i < src_dims.size(); ++i) {
    memset_size *= src_dims[i];
  }
  const size_t& memset_bytes = memset_size * sizeof(T);

#ifdef PADDLE_WITH_HIP
  hipMemset(p_output, 0, memset_bytes);
#else
  cudaMemset(p_output, 0, memset_bytes);
#endif

  if (index_size == 0) return;

  int64_t slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) {
    slice_size *= src_dims[i];
  }
  const T* p_src = out_grad.data<T>();
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();

#ifdef PADDLE_WITH_HIP
  int block = 256;
#else
  int block = 1024;
#endif
  int64_t n = slice_size * index_size;
  int64_t max_grid_dimx = ctx.GetCUDAMaxGridDimSize()[0];
  int64_t grid_tmp = (n + block - 1) / block;
  int64_t grid = grid_tmp < max_grid_dimx ? grid_tmp : max_grid_dimx;
  int64_t input_size = src_dims[0];
  if (pool_type == "SUM") {
    GraphSendRecvSumCUDAFunctor<T, IndexT> functor;
    GraphSendRecvCUDAKernel<
        T,
        IndexT,
        GraphSendRecvSumCUDAFunctor<T,
                                    IndexT>><<<grid, block, 0, ctx.stream()>>>(
        p_src, d_index, s_index, p_output, index_size, slice_size, functor);
  } else if (pool_type == "MEAN") {
    const int32_t* s_count = dst_count->data<int32_t>();
    ManipulateMeanGradCUDAKernel<T, IndexT><<<grid, block, 0, ctx.stream()>>>(
        p_src, d_index, s_index, p_output, index_size, slice_size, s_count);
  } else if (pool_type == "MAX" || pool_type == "MIN") {
    const T* ptr_input = x.data<T>();
    const T* ptr_output = out->data<T>();
    ManipulateMinMaxGradCUDAKernel<T, IndexT><<<grid, block, 0, ctx.stream()>>>(
        p_src,
        d_index,
        s_index,
        p_output,
        index_size,
        slice_size,
        ptr_input,
        ptr_output);
  }
}

template <typename T, typename Context>
void GraphSendRecvGradKernel(const Context& ctx,
                             const DenseTensor& x,
                             const DenseTensor& src_index,
                             const DenseTensor& dst_index,
                             paddle::optional<const DenseTensor&> out,
                             paddle::optional<const DenseTensor&> dst_count,
                             const DenseTensor& out_grad,
                             const std::string& pool_type,
                             DenseTensor* x_grad) {
  auto index_type = src_index.dtype();
  if (index_type == phi::DataType::INT32) {
    GraphSendRecvGradOpCUDAKernelLaunchHelper<Context, T, int32_t>(
        ctx,
        out_grad,
        x,
        src_index,
        dst_index,
        pool_type,
        x_grad,
        dst_count.get_ptr(),
        out.get_ptr());
  } else if (index_type == phi::DataType::INT64) {
    GraphSendRecvGradOpCUDAKernelLaunchHelper<Context, T, int64_t>(
        ctx,
        out_grad,
        x,
        src_index,
        dst_index,
        pool_type,
        x_grad,
        dst_count.get_ptr(),
        out.get_ptr());
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(graph_send_recv_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::GraphSendRecvGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
