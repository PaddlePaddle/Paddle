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
#include "paddle/phi/kernels/graph_send_recv_kernel.h"

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <algorithm>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename Context, typename T, typename IndexT>
void GraphSendRecvOpCUDAKernelLaunchHelper(const Context& ctx,
                                           const DenseTensor& x,
                                           const DenseTensor& src_index,
                                           const DenseTensor& dst_index,
                                           const std::string& pool_type,
                                           int64_t out_size,
                                           DenseTensor* out,
                                           DenseTensor* dst_count = nullptr) {
  const int& index_size = src_index.dims()[0];
  ctx.template Alloc<T>(out);
  T* p_output = out->data<T>();
  const auto& src_dims = x.dims();
  int64_t memset_size = 1;
  if (out_size <= 0) {
    for (int i = 0; i < src_dims.size(); ++i) {
      memset_size *= src_dims[i];
    }
  } else {
    memset_size = out_size;
    for (int i = 1; i < src_dims.size(); ++i) {
      memset_size *= src_dims[i];
    }
  }
  const size_t& memset_bytes = memset_size * sizeof(T);
  if (pool_type == "SUM" || pool_type == "MEAN") {
#ifdef PADDLE_WITH_HIP
    hipMemset(p_output, 0, memset_bytes);
#else
    cudaMemset(p_output, 0, memset_bytes);
#endif
  } else if (pool_type == "MAX") {
    thrust::device_ptr<T> p_output_ptr(p_output);
    thrust::fill(thrust::device,
                 p_output_ptr,
                 p_output_ptr + memset_size,
                 std::numeric_limits<T>::min());
  } else if (pool_type == "MIN") {
    thrust::device_ptr<T> p_output_ptr(p_output);
    thrust::fill(thrust::device,
                 p_output_ptr,
                 p_output_ptr + memset_size,
                 std::numeric_limits<T>::max());
  }

  if (index_size == 0) return;

  int64_t slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) {
    slice_size *= src_dims[i];
  }
  const T* p_src = x.data<T>();
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
        p_src, s_index, d_index, p_output, index_size, slice_size, functor);
  } else if (pool_type == "MAX") {
    GraphSendRecvMaxCUDAFunctor<T, IndexT> functor;
    GraphSendRecvCUDAKernel<
        T,
        IndexT,
        GraphSendRecvMaxCUDAFunctor<T,
                                    IndexT>><<<grid, block, 0, ctx.stream()>>>(
        p_src, s_index, d_index, p_output, index_size, slice_size, functor);

    if (out_size > 0) {
      input_size = out_size;
    }
    int64_t grid_max_tmp = (input_size * slice_size + block - 1) / block;
    int64_t grid_max =
        grid_max_tmp < max_grid_dimx ? grid_max_tmp : max_grid_dimx;
    InputResetMaxCUDAKernel<T><<<grid_max, block, 0, ctx.stream()>>>(
        p_output, input_size, slice_size);
  } else if (pool_type == "MIN") {
    GraphSendRecvMinCUDAFunctor<T, IndexT> functor;
    GraphSendRecvCUDAKernel<
        T,
        IndexT,
        GraphSendRecvMinCUDAFunctor<T,
                                    IndexT>><<<grid, block, 0, ctx.stream()>>>(
        p_src, s_index, d_index, p_output, index_size, slice_size, functor);

    if (out_size > 0) {
      input_size = out_size;
    }
    int64_t grid_min_tmp = (input_size * slice_size + block - 1) / block;
    int64_t grid_min =
        grid_min_tmp < max_grid_dimx ? grid_min_tmp : max_grid_dimx;
    InputResetMinCUDAKernel<T><<<grid_min, block, 0, ctx.stream()>>>(
        p_output, input_size, slice_size);
  } else if (pool_type == "MEAN") {
    GraphSendRecvSumCUDAFunctor<T, IndexT> functor;
    GraphSendRecvCUDAKernel<
        T,
        IndexT,
        GraphSendRecvSumCUDAFunctor<T,
                                    IndexT>><<<grid, block, 0, ctx.stream()>>>(
        p_src, s_index, d_index, p_output, index_size, slice_size, functor);

    ctx.template Alloc<int32_t>(dst_count);
    int32_t* p_dst_count = dst_count->data<int32_t>();
    if (out_size > 0) {
      input_size = out_size;
    }

#ifdef PADDLE_WITH_HIP
    hipMemset(p_dst_count, 0, input_size * sizeof(int));
#else
    cudaMemset(p_dst_count, 0, input_size * sizeof(int));
#endif

    int64_t grid_count = (index_size + block - 1) / block;
    ComputeCountCUDAKernel<T, IndexT><<<grid_count, block, 0, ctx.stream()>>>(
        p_dst_count, d_index, index_size);

    int64_t grid_mean_tmp = (input_size * slice_size + block - 1) / block;
    int64_t grid_mean =
        grid_mean_tmp < max_grid_dimx ? grid_mean_tmp : max_grid_dimx;
    ManipulateMeanCUDAKernel<T><<<grid_mean, block, 0, ctx.stream()>>>(
        p_output, p_dst_count, input_size, slice_size);
  }
}

template <typename T, typename Context>
void GraphSendRecvKernel(const Context& ctx,
                         const DenseTensor& x,
                         const DenseTensor& src_index,
                         const DenseTensor& dst_index,
                         const std::string& pool_type,
                         int64_t out_size,
                         DenseTensor* out,
                         DenseTensor* dst_count) {
  auto index_type = src_index.dtype();
  if (index_type == phi::DataType::INT32) {
    GraphSendRecvOpCUDAKernelLaunchHelper<Context, T, int32_t>(
        ctx, x, src_index, dst_index, pool_type, out_size, out, dst_count);
  } else if (index_type == phi::DataType::INT64) {
    GraphSendRecvOpCUDAKernelLaunchHelper<Context, T, int64_t>(
        ctx, x, src_index, dst_index, pool_type, out_size, out, dst_count);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(graph_send_recv,
                   GPU,
                   ALL_LAYOUT,
                   phi::GraphSendRecvKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
