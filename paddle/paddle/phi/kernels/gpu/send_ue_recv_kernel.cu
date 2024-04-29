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

#include "paddle/phi/kernels/send_ue_recv_kernel.h"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <algorithm>
#include <vector>

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/graph_send_recv_funcs.h"
#include "paddle/phi/kernels/gpu/graph_send_ue_recv_funcs.h"
#include "paddle/phi/kernels/impl/graph_message_passing_impl.h"

namespace phi {

template <typename Context, typename T, typename IndexT>
void GraphSendUERecvOpCUDAKernelLaunchHelper(const Context& ctx,
                                             const DenseTensor& x,
                                             const DenseTensor& e,
                                             const DenseTensor& src_index,
                                             const DenseTensor& dst_index,
                                             const std::string& message_op,
                                             const std::string& reduce_op,
                                             int64_t out_size,
                                             DenseTensor* out,
                                             DenseTensor* dst_count = nullptr) {
  const int& index_size = src_index.dims()[0];
  auto out_dims = out->dims();
  int64_t memset_size = 1;
  std::vector<int64_t> dims_ = common::vectorize(out_dims);
  if (out_size <= 0) {
    dims_[0] = x.dims()[0];
  } else {
    dims_[0] = out_size;
  }
  out->Resize(common::make_ddim(dims_));
  for (size_t i = 0; i < dims_.size(); i++) {
    memset_size *= dims_[i];
  }

  ctx.template Alloc<T>(out);
  T* out_data = out->data<T>();
  const size_t& memset_bytes = memset_size * sizeof(T);
  funcs::SetConstant<Context, T> constant_functor;
  if (reduce_op == "SUM" || reduce_op == "MEAN") {
    constant_functor(ctx, out, static_cast<T>(0));
  } else if (reduce_op == "MAX") {
    constant_functor(ctx, out, std::numeric_limits<T>::lowest());

  } else if (reduce_op == "MIN") {
    constant_functor(ctx, out, std::numeric_limits<T>::max());
  }

  if (index_size == 0) return;

  const auto& bcast_info = phi::CalcBCastInfo(x.dims(), e.dims());

  const T* x_data = x.data<T>();
  const T* e_data = e.data<T>();
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();

  thrust::device_vector<int64_t> x_bcastoff, e_bcastoff;
  if (bcast_info.use_bcast) {
    CopyBCastOff(bcast_info, &x_bcastoff, &e_bcastoff);
  }

  int64_t out_len = bcast_info.out_len;
  const int ntx = FindNumThreads(out_len, ctx.GetMaxThreadsPerBlock());
  const int nty = ctx.GetMaxThreadsPerBlock() / ntx;
  const int nbx = (out_len + ntx - 1) / ntx;
  const int nby = FindNumBlocks('y', (index_size + nty - 1) / nty);
  const dim3 grid(nbx, nby);
  const dim3 block(ntx, nty);
  int64_t input_size = x.dims()[0];
  int block_ = 1024;
  if (reduce_op == "SUM" || reduce_op == "MEAN") {
    GraphSendUERecvSumCUDAFunctor<T> sum_functor;
    if (message_op == "ADD") {
      funcs::AddFunctor<T> add_funtor;
      GraphSendUERecvCUDAKernel<T,
                                IndexT,
                                GraphSendUERecvSumCUDAFunctor<T>,
                                funcs::AddFunctor<T>>
          <<<grid, block, 0, ctx.stream()>>>(
              x_data,
              e_data,
              s_index,
              d_index,
              thrust::raw_pointer_cast(x_bcastoff.data()),
              thrust::raw_pointer_cast(e_bcastoff.data()),
              out_data,
              index_size,
              bcast_info.l_len,
              bcast_info.r_len,
              out_len,
              bcast_info.use_bcast,
              add_funtor,
              sum_functor);
    } else if (message_op == "MUL") {
      funcs::MultiplyFunctor<T> mul_functor;
      GraphSendUERecvCUDAKernel<T,
                                IndexT,
                                GraphSendUERecvSumCUDAFunctor<T>,
                                funcs::MultiplyFunctor<T>>
          <<<grid, block, 0, ctx.stream()>>>(
              x_data,
              e_data,
              s_index,
              d_index,
              thrust::raw_pointer_cast(x_bcastoff.data()),
              thrust::raw_pointer_cast(e_bcastoff.data()),
              out_data,
              index_size,
              bcast_info.l_len,
              bcast_info.r_len,
              out_len,
              bcast_info.use_bcast,
              mul_functor,
              sum_functor);
    }
    if (reduce_op == "MEAN") {
      input_size = out_size <= 0 ? x.dims()[0] : out_size;
      dst_count->Resize({input_size});
      ctx.template Alloc<int>(dst_count);
      int* dst_count_data = dst_count->data<int>();
#ifdef PADDLE_WITH_HIP
      hipMemset(dst_count_data, 0, input_size * sizeof(int));
#else
      cudaMemsetAsync(
          dst_count_data, 0, input_size * sizeof(int), ctx.stream());
#endif
      int64_t grid_count = (index_size + block_ - 1) / block_;
      ComputeCountCUDAKernel<T, IndexT>
          <<<grid_count, block_, 0, ctx.stream()>>>(
              dst_count_data, d_index, index_size);

      int64_t grid_mean = (input_size * out_len + block_ - 1) / block_;
      int64_t max_grid_dimx = ctx.GetCUDAMaxGridDimSize()[0];
      int64_t grid_mean_ =
          grid_mean < max_grid_dimx ? grid_mean : max_grid_dimx;
      ManipulateMeanCUDAKernel<T><<<grid_mean_, block_, 0, ctx.stream()>>>(
          out_data, dst_count_data, input_size, out_len);
    }
  } else if (reduce_op == "MAX") {
    GraphSendUERecvMaxCUDAFunctor<T> max_functor;
    if (message_op == "ADD") {
      funcs::AddFunctor<T> add_funtor;
      GraphSendUERecvCUDAKernel<T,
                                IndexT,
                                GraphSendUERecvMaxCUDAFunctor<T>,
                                funcs::AddFunctor<T>>
          <<<grid, block, 0, ctx.stream()>>>(
              x_data,
              e_data,
              s_index,
              d_index,
              thrust::raw_pointer_cast(x_bcastoff.data()),
              thrust::raw_pointer_cast(e_bcastoff.data()),
              out_data,
              index_size,
              bcast_info.l_len,
              bcast_info.r_len,
              out_len,
              bcast_info.use_bcast,
              add_funtor,
              max_functor);
    } else if (message_op == "MUL") {
      funcs::MultiplyFunctor<T> mul_functor;
      GraphSendUERecvCUDAKernel<T,
                                IndexT,
                                GraphSendUERecvMaxCUDAFunctor<T>,
                                funcs::MultiplyFunctor<T>>
          <<<grid, block, 0, ctx.stream()>>>(
              x_data,
              e_data,
              s_index,
              d_index,
              thrust::raw_pointer_cast(x_bcastoff.data()),
              thrust::raw_pointer_cast(e_bcastoff.data()),
              out_data,
              index_size,
              bcast_info.l_len,
              bcast_info.r_len,
              out_len,
              bcast_info.use_bcast,
              mul_functor,
              max_functor);
    }
    if (out_size > 0) {
      input_size = out_size;
    }
    int64_t grid_max = (input_size * out_len + block_ - 1) / block_;
    int64_t max_grid_dimx = ctx.GetCUDAMaxGridDimSize()[0];
    int64_t grid_max_ = grid_max < max_grid_dimx ? grid_max : max_grid_dimx;
    InputResetMaxCUDAKernel<T>
        <<<grid_max_, block_, 0, ctx.stream()>>>(out_data, input_size, out_len);
  } else if (reduce_op == "MIN") {
    GraphSendUERecvMinCUDAFunctor<T> min_functor;
    if (message_op == "ADD") {
      funcs::AddFunctor<T> add_funtor;
      GraphSendUERecvCUDAKernel<T,
                                IndexT,
                                GraphSendUERecvMinCUDAFunctor<T>,
                                funcs::AddFunctor<T>>
          <<<grid, block, 0, ctx.stream()>>>(
              x_data,
              e_data,
              s_index,
              d_index,
              thrust::raw_pointer_cast(x_bcastoff.data()),
              thrust::raw_pointer_cast(e_bcastoff.data()),
              out_data,
              index_size,
              bcast_info.l_len,
              bcast_info.r_len,
              out_len,
              bcast_info.use_bcast,
              add_funtor,
              min_functor);
    } else if (message_op == "MUL") {
      funcs::MultiplyFunctor<T> mul_functor;
      GraphSendUERecvCUDAKernel<T,
                                IndexT,
                                GraphSendUERecvMinCUDAFunctor<T>,
                                funcs::MultiplyFunctor<T>>
          <<<grid, block, 0, ctx.stream()>>>(
              x_data,
              e_data,
              s_index,
              d_index,
              thrust::raw_pointer_cast(x_bcastoff.data()),
              thrust::raw_pointer_cast(e_bcastoff.data()),
              out_data,
              index_size,
              bcast_info.l_len,
              bcast_info.r_len,
              out_len,
              bcast_info.use_bcast,
              mul_functor,
              min_functor);
    }
    if (out_size > 0) {
      input_size = out_size;
    }
    int64_t grid_min = (input_size * out_len + block_ - 1) / block_;
    int64_t max_grid_dimx = ctx.GetCUDAMaxGridDimSize()[0];
    int64_t grid_min_ = grid_min < max_grid_dimx ? grid_min : max_grid_dimx;
    InputResetMinCUDAKernel<T>
        <<<grid_min_, block_, 0, ctx.stream()>>>(out_data, input_size, out_len);
  }
}

template <typename T, typename Context>
void SendUERecvKernel(const Context& ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& src_index,
                      const DenseTensor& dst_index,
                      const std::string& message_op,
                      const std::string& reduce_op,
                      const IntArray& out_size,
                      DenseTensor* out,
                      DenseTensor* dst_count) {
  auto index_type = src_index.dtype();
  auto& out_size_data = out_size.GetData();
  if (index_type == phi::DataType::INT32) {
    GraphSendUERecvOpCUDAKernelLaunchHelper<Context, T, int32_t>(
        ctx,
        x,
        y,
        src_index,
        dst_index,
        message_op,
        reduce_op,
        out_size_data[0],
        out,
        dst_count);
  } else if (index_type == phi::DataType::INT64) {
    GraphSendUERecvOpCUDAKernelLaunchHelper<Context, T, int64_t>(
        ctx,
        x,
        y,
        src_index,
        dst_index,
        message_op,
        reduce_op,
        out_size_data[0],
        out,
        dst_count);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(send_ue_recv,
                   GPU,
                   ALL_LAYOUT,
                   phi::SendUERecvKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
}
