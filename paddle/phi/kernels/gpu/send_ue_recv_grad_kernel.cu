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

#include "paddle/phi/kernels/send_ue_recv_grad_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/graph_send_recv_funcs.h"
#include "paddle/phi/kernels/gpu/graph_send_ue_recv_funcs.h"
#include "paddle/phi/kernels/impl/graph_message_passing_impl.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename Context, typename T, typename IndexT>
void CalculateXEGradForMinMax(const Context& ctx,
                              const T* out_grad,
                              const T* x_data,
                              const T* e_data,
                              const phi::DDim& x_dims,
                              const phi::DDim& e_dims,
                              const IndexT* s_index,
                              const IndexT* d_index,
                              const std::string& message_op,
                              const std::string& reduce_op,
                              int64_t index_size,
                              T* x_grad,
                              T* e_grad,
                              const DenseTensor* out = nullptr) {
  const T* out_data = out->data<T>();
  const auto& bcast_info = phi::CalcBCastInfo(x_dims, e_dims);
  thrust::device_vector<int64_t> l_bcastoff, r_bcastoff;
  if (bcast_info.use_bcast) {
    CopyBCastOff(bcast_info, &l_bcastoff, &r_bcastoff);
  }

  int64_t out_len = bcast_info.out_len;
  const int ntx = FindNumThreads(out_len, ctx.GetMaxThreadsPerBlock());
  const int nty = ctx.GetMaxThreadsPerBlock() / ntx;
  const int nbx = (out_len + ntx - 1) / ntx;
  const int nby = FindNumBlocks('y', (index_size + nty - 1) / nty);
  const dim3 grid(nbx, nby);
  const dim3 block(ntx, nty);

  if (message_op == "ADD") {
    ManipulateMinMaxGradCUDAKernelForAdd<T, IndexT>
        <<<grid, block, 0, ctx.stream()>>>(
            x_data,
            e_data,
            out_data,
            out_grad,
            d_index,
            s_index,
            thrust::raw_pointer_cast(l_bcastoff.data()),
            thrust::raw_pointer_cast(r_bcastoff.data()),
            x_grad,
            e_grad,
            index_size,
            bcast_info.l_len,
            bcast_info.r_len,
            out_len,
            bcast_info.use_bcast);
  } else if (message_op == "MUL") {
    ManipulateMinMaxGradCUDAKernelForMul<T, IndexT>
        <<<grid, block, 0, ctx.stream()>>>(
            x_data,
            e_data,
            out_data,
            out_grad,
            d_index,
            s_index,
            thrust::raw_pointer_cast(l_bcastoff.data()),
            thrust::raw_pointer_cast(r_bcastoff.data()),
            x_grad,
            e_grad,
            index_size,
            bcast_info.l_len,
            bcast_info.r_len,
            out_len,
            bcast_info.use_bcast);
  }
}

template <typename Context, typename T, typename IndexT>
void CalculateXGrad(const Context& ctx,
                    const T* out_grad,
                    const T* x_data,
                    const T* e_data,
                    const phi::DDim& out_grad_dims,
                    const phi::DDim& x_dims,
                    const phi::DDim& e_dims,
                    const IndexT* s_index,
                    const IndexT* d_index,
                    const std::string& message_op,
                    const std::string& reduce_op,
                    int64_t index_size,
                    int64_t slice_size,
                    T* x_grad,
                    const DenseTensor& out_grad_tensor,
                    const DenseTensor* dst_count = nullptr,
                    const DenseTensor* out = nullptr) {
#ifdef PADDLE_WITH_HIP
  int block = 256;
#else
  int block = 1024;
#endif
  int64_t n = slice_size * index_size;
  int max_grid_dimx = ctx.GetCUDAMaxGridDimSize()[0];
  int64_t grid_tmp = (n + block - 1) / block;
  int64_t grid = grid_tmp < max_grid_dimx ? grid_tmp : max_grid_dimx;
  std::vector<int64_t> reduce_idx;
  bool reduce = ReduceGrad(out_grad_dims, x_dims, reduce_idx);
  if (reduce_op == "SUM") {
    if (message_op == "ADD") {
      GraphSendRecvSumCUDAFunctor<T, IndexT> functor;
      if (!reduce) {
        GraphSendRecvCUDAKernel<T,
                                IndexT,
                                GraphSendRecvSumCUDAFunctor<T, IndexT>>
            <<<grid, block, 0, ctx.stream()>>>(out_grad,
                                               d_index,
                                               s_index,
                                               x_grad,
                                               index_size,
                                               slice_size,
                                               functor);
      } else {
        const auto& bcast_info = phi::CalcBCastInfo(out_grad_dims, e_dims);
        DenseTensor x_grad_v2 =
            phi::EmptyLike<T, Context>(ctx, out_grad_tensor);
        phi::funcs::SetConstant<Context, T>()(ctx, &x_grad_v2, T(0));
        T* x_grad_v2_data = x_grad_v2.data<T>();
        GraphSendRecvCUDAKernel<T,
                                IndexT,
                                GraphSendRecvSumCUDAFunctor<T, IndexT>>
            <<<grid, block, 0, ctx.stream()>>>(out_grad,
                                               d_index,
                                               s_index,
                                               x_grad_v2_data,
                                               index_size,
                                               bcast_info.out_len,
                                               functor);
        // Run reduce_sum
        DenseTensor x_grad_out = phi::Sum<T, Context>(
            ctx,
            x_grad_v2,
            phi::IntArray(reduce_idx),
            paddle::experimental::CppTypeToDataType<T>::Type(),
            true);
#ifdef PADDLE_WITH_HIP
        hipMemcpy(x_grad,
                  x_grad_out.data<T>(),
                  x_grad_out.numel() * sizeof(T),
                  hipMemcpyDeviceToDevice);
#else
        cudaMemcpy(x_grad,
                   x_grad_out.data<T>(),
                   x_grad_out.numel() * sizeof(T),
                   cudaMemcpyDeviceToDevice);
#endif
      }
    } else if (message_op == "MUL") {
      const auto& bcast_info = phi::CalcBCastInfo(out_grad_dims, e_dims);
      thrust::device_vector<int64_t> l_bcastoff, r_bcastoff;
      if (bcast_info.use_bcast) {
        CopyBCastOff(bcast_info, &l_bcastoff, &r_bcastoff);
      }
      int64_t out_len = bcast_info.out_len;
      const int ntx = FindNumThreads(out_len, ctx.GetMaxThreadsPerBlock());
      const int nty = ctx.GetMaxThreadsPerBlock() / ntx;
      const int nbx = (out_len + ntx - 1) / ntx;
      const int nby = FindNumBlocks('y', (index_size + nty - 1) / nty);
      const dim3 grid_(nbx, nby);
      const dim3 block_(ntx, nty);
      funcs::MultiplyFunctor<T> mul_functor;
      GraphSendUERecvSumCUDAFunctor<T> sum_functor;
      if (!reduce) {
        GraphSendUERecvCUDAKernel<T,
                                  IndexT,
                                  GraphSendUERecvSumCUDAFunctor<T>,
                                  funcs::MultiplyFunctor<T>>
            <<<grid_, block_, 0, ctx.stream()>>>(
                out_grad,
                e_data,
                d_index,
                s_index,
                thrust::raw_pointer_cast(l_bcastoff.data()),
                thrust::raw_pointer_cast(r_bcastoff.data()),
                x_grad,
                index_size,
                bcast_info.l_len,
                bcast_info.r_len,
                out_len,
                bcast_info.use_bcast,
                mul_functor,
                sum_functor);
      } else {
        DenseTensor x_grad_v2 =
            phi::EmptyLike<T, Context>(ctx, out_grad_tensor);
        phi::funcs::SetConstant<Context, T>()(ctx, &x_grad_v2, T(0));
        T* x_grad_v2_data = x_grad_v2.data<T>();
        GraphSendUERecvCUDAKernel<T,
                                  IndexT,
                                  GraphSendUERecvSumCUDAFunctor<T>,
                                  funcs::MultiplyFunctor<T>>
            <<<grid_, block_, 0, ctx.stream()>>>(
                out_grad,
                e_data,
                d_index,
                s_index,
                thrust::raw_pointer_cast(l_bcastoff.data()),
                thrust::raw_pointer_cast(r_bcastoff.data()),
                x_grad_v2_data,
                index_size,
                bcast_info.l_len,
                bcast_info.r_len,
                out_len,
                bcast_info.use_bcast,
                mul_functor,
                sum_functor);
        DenseTensor x_grad_out = phi::Sum<T, Context>(
            ctx,
            x_grad_v2,
            phi::IntArray(reduce_idx),
            paddle::experimental::CppTypeToDataType<T>::Type(),
            true);
#ifdef PADDLE_WITH_HIP
        hipMemcpy(x_grad,
                  x_grad_out.data<T>(),
                  x_grad_out.numel() * sizeof(T),
                  hipMemcpyDeviceToDevice);
#else
        cudaMemcpy(x_grad,
                   x_grad_out.data<T>(),
                   x_grad_out.numel() * sizeof(T),
                   cudaMemcpyDeviceToDevice);
#endif
      }
    }
  } else if (reduce_op == "MEAN") {
    const int* s_count = dst_count->data<int>();
    if (message_op == "ADD") {
      if (!reduce) {
        ManipulateMeanGradCUDAKernel<T, IndexT>
            <<<grid, block, 0, ctx.stream()>>>(out_grad,
                                               d_index,
                                               s_index,
                                               x_grad,
                                               index_size,
                                               slice_size,
                                               s_count);
      } else {
        const auto& bcast_info = phi::CalcBCastInfo(out_grad_dims, e_dims);
        DenseTensor x_grad_v2 =
            phi::EmptyLike<T, Context>(ctx, out_grad_tensor);
        phi::funcs::SetConstant<Context, T>()(ctx, &x_grad_v2, T(0));
        T* x_grad_v2_data = x_grad_v2.data<T>();
        ManipulateMeanGradCUDAKernel<T, IndexT>
            <<<grid, block, 0, ctx.stream()>>>(out_grad,
                                               d_index,
                                               s_index,
                                               x_grad_v2_data,
                                               index_size,
                                               bcast_info.out_len,
                                               s_count);
        // Run reduce_sum
        DenseTensor x_grad_out = phi::Sum<T, Context>(
            ctx,
            x_grad_v2,
            phi::IntArray(reduce_idx),
            paddle::experimental::CppTypeToDataType<T>::Type(),
            true);
#ifdef PADDLE_WITH_HIP
        hipMemcpy(x_grad,
                  x_grad_out.data<T>(),
                  x_grad_out.numel() * sizeof(T),
                  hipMemcpyDeviceToDevice);
#else
        cudaMemcpy(x_grad,
                   x_grad_out.data<T>(),
                   x_grad_out.numel() * sizeof(T),
                   cudaMemcpyDeviceToDevice);
#endif
      }
    } else if (message_op == "MUL") {
      const auto& bcast_info = phi::CalcBCastInfo(out_grad_dims, e_dims);
      thrust::device_vector<int64_t> l_bcastoff, r_bcastoff;
      if (bcast_info.use_bcast) {
        CopyBCastOff(bcast_info, &l_bcastoff, &r_bcastoff);
      }
      int64_t out_len = bcast_info.out_len;
      const int ntx = FindNumThreads(out_len, ctx.GetMaxThreadsPerBlock());
      const int nty = ctx.GetMaxThreadsPerBlock() / ntx;
      const int nbx = (out_len + ntx - 1) / ntx;
      const int nby = FindNumBlocks('y', (index_size + nty - 1) / nty);
      const dim3 grid_(nbx, nby);
      const dim3 block_(ntx, nty);
      if (!reduce) {
        ManipulateMeanGradCUDAKernelForMulX<T, IndexT>
            <<<grid_, block_, 0, ctx.stream()>>>(
                out_grad,
                e_data,
                d_index,
                s_index,
                s_count,
                thrust::raw_pointer_cast(l_bcastoff.data()),
                thrust::raw_pointer_cast(r_bcastoff.data()),
                x_grad,
                index_size,
                bcast_info.l_len,
                bcast_info.r_len,
                out_len,
                bcast_info.use_bcast);
      } else {
        DenseTensor x_grad_v2 =
            phi::EmptyLike<T, Context>(ctx, out_grad_tensor);
        phi::funcs::SetConstant<Context, T>()(ctx, &x_grad_v2, T(0));
        T* x_grad_v2_data = x_grad_v2.data<T>();
        ManipulateMeanGradCUDAKernelForMulX<T, IndexT>
            <<<grid_, block_, 0, ctx.stream()>>>(
                out_grad,
                e_data,
                d_index,
                s_index,
                s_count,
                thrust::raw_pointer_cast(l_bcastoff.data()),
                thrust::raw_pointer_cast(r_bcastoff.data()),
                x_grad_v2_data,
                index_size,
                bcast_info.l_len,
                bcast_info.r_len,
                out_len,
                bcast_info.use_bcast);
        // Run reduce_sum
        DenseTensor x_grad_out = phi::Sum<T, Context>(
            ctx,
            x_grad_v2,
            phi::IntArray(reduce_idx),
            paddle::experimental::CppTypeToDataType<T>::Type(),
            true);
        // TODO(daisiming): Whether use x_grad instead.
#ifdef PADDLE_WITH_HIP
        hipMemcpy(x_grad,
                  x_grad_out.data<T>(),
                  x_grad_out.numel() * sizeof(T),
                  hipMemcpyDeviceToDevice);
#else
        cudaMemcpy(x_grad,
                   x_grad_out.data<T>(),
                   x_grad_out.numel() * sizeof(T),
                   cudaMemcpyDeviceToDevice);
#endif
      }
    }
  }
}

template <typename Context, typename T, typename IndexT>
void CalculateEGrad(const Context& ctx,
                    const T* out_grad,
                    const T* x_data,
                    const T* e_data,
                    const phi::DDim& x_dims,
                    const phi::DDim& e_dims,
                    const IndexT* s_index,
                    const IndexT* d_index,
                    const std::string& message_op,
                    const std::string& reduce_op,
                    int64_t index_size,
                    T* e_grad,
                    const DenseTensor* dst_count = nullptr) {
  const auto& bcast_info = phi::CalcBCastInfo(x_dims, e_dims);
  thrust::device_vector<int64_t> l_bcastoff, r_bcastoff;
  if (bcast_info.use_bcast) {
    CopyBCastOff(bcast_info, &l_bcastoff, &r_bcastoff);
  }
  int64_t out_len = bcast_info.out_len;
  const int ntx = FindNumThreads(out_len, ctx.GetMaxThreadsPerBlock());
  const int nty = ctx.GetMaxThreadsPerBlock() / ntx;
  const int nbx = (out_len + ntx - 1) / ntx;
  const int nby = FindNumBlocks('y', (index_size + nty - 1) / nty);
  const dim3 grid(nbx, nby);
  const dim3 block(ntx, nty);
  if (reduce_op == "SUM") {
    if (message_op == "ADD") {
      ManipulateSumGradCUDAKernelForAddE<T, IndexT>
          <<<grid, block, 0, ctx.stream()>>>(
              out_grad,
              d_index,
              thrust::raw_pointer_cast(r_bcastoff.data()),
              e_grad,
              index_size,
              bcast_info.r_len,
              out_len,
              bcast_info.use_bcast);
    } else if (message_op == "MUL") {
      ManipulateSumGradCUDAKernelForMulE<T, IndexT>
          <<<grid, block, 0, ctx.stream()>>>(
              x_data,
              out_grad,
              s_index,
              d_index,
              thrust::raw_pointer_cast(l_bcastoff.data()),
              thrust::raw_pointer_cast(r_bcastoff.data()),
              e_grad,
              index_size,
              bcast_info.l_len,
              bcast_info.r_len,
              out_len,
              bcast_info.use_bcast);
    }
  } else if (reduce_op == "MEAN") {
    const int* s_count = dst_count->data<int>();
    if (message_op == "ADD") {
      ManipulateMeanGradCUDAKernelForAddE<T, IndexT>
          <<<grid, block, 0, ctx.stream()>>>(
              out_grad,
              d_index,
              s_count,
              thrust::raw_pointer_cast(r_bcastoff.data()),
              e_grad,
              index_size,
              bcast_info.r_len,
              out_len,
              bcast_info.use_bcast);
    } else if (message_op == "MUL") {
      ManipulateMeanGradCUDAKernelForMulE<T, IndexT>
          <<<grid, block, 0, ctx.stream()>>>(
              x_data,
              out_grad,
              s_index,
              d_index,
              s_count,
              thrust::raw_pointer_cast(l_bcastoff.data()),
              thrust::raw_pointer_cast(r_bcastoff.data()),
              e_grad,
              index_size,
              bcast_info.l_len,
              bcast_info.r_len,
              out_len,
              bcast_info.use_bcast);
    }
  }
}

template <typename Context, typename T, typename IndexT>
void GraphSendUERecvGradOpCUDAKernelLaunchHelper(
    const Context& ctx,
    const DenseTensor& out_grad,
    const DenseTensor& x,
    const DenseTensor& e,
    const DenseTensor& src_index,
    const DenseTensor& dst_index,
    const std::string& message_op,
    const std::string& reduce_op,
    DenseTensor* x_grad,
    DenseTensor* e_grad,
    const DenseTensor* dst_count = nullptr,
    const DenseTensor* out = nullptr) {
  const int& index_size = dst_index.dims()[0];

  ctx.template Alloc<T>(x_grad);
  T* x_grad_data = x_grad->data<T>();
  ctx.template Alloc<T>(e_grad);
  T* e_grad_data = e_grad->data<T>();
  const auto& x_dims = x.dims();
  const auto& e_dims = e.dims();
  int64_t memset_size_x = 1, memset_size_e = 1;
  int64_t slice_size = 1;
  for (int i = 0; i < x_dims.size(); i++) {
    memset_size_x *= x_dims[i];
    if (i > 0) slice_size *= x_dims[i];
  }
  for (int i = 0; i < e_dims.size(); i++) {
    memset_size_e *= e_dims[i];
  }
  const size_t& memset_bytes_x = memset_size_x * sizeof(T);
  const size_t& memset_bytes_e = memset_size_e * sizeof(T);
#ifdef PADDLE_WITH_HIP
  hipMemset(x_grad_data, 0, memset_bytes_x);
  hipMemset(e_grad_data, 0, memset_bytes_e);
#else
  cudaMemset(x_grad_data, 0, memset_bytes_x);
  cudaMemset(e_grad_data, 0, memset_bytes_e);
#endif

  if (index_size == 0) return;

  const T* out_grad_data = out_grad.data<T>();
  const T* x_data = x.data<T>();
  const T* e_data = e.data<T>();
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();

  if (reduce_op == "SUM" || reduce_op == "MEAN") {
    CalculateXGrad<Context, T, IndexT>(ctx,
                                       out_grad_data,
                                       x_data,
                                       e_data,
                                       out_grad.dims(),
                                       x_dims,
                                       e_dims,
                                       s_index,
                                       d_index,
                                       message_op,
                                       reduce_op,
                                       index_size,
                                       slice_size,
                                       x_grad_data,
                                       out_grad,
                                       dst_count,
                                       out);
    CalculateEGrad<Context, T, IndexT>(ctx,
                                       out_grad_data,
                                       x_data,
                                       e_data,
                                       x_dims,
                                       e_dims,
                                       s_index,
                                       d_index,
                                       message_op,
                                       reduce_op,
                                       index_size,
                                       e_grad_data,
                                       dst_count);
  } else if (reduce_op == "MIN" || reduce_op == "MAX") {
    CalculateXEGradForMinMax<Context, T, IndexT>(ctx,
                                                 out_grad_data,
                                                 x_data,
                                                 e_data,
                                                 x_dims,
                                                 e_dims,
                                                 s_index,
                                                 d_index,
                                                 message_op,
                                                 reduce_op,
                                                 index_size,
                                                 x_grad_data,
                                                 e_grad_data,
                                                 out);
  }
}

template <typename T, typename Context>
void SendUERecvGradKernel(const Context& ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          const DenseTensor& src_index,
                          const DenseTensor& dst_index,
                          const paddle::optional<DenseTensor>& out,
                          const paddle::optional<DenseTensor>& dst_count,
                          const DenseTensor& out_grad,
                          const std::string& message_op,
                          const std::string& reduce_op,
                          DenseTensor* x_grad,
                          DenseTensor* y_grad) {
  auto index_type = src_index.dtype();
  if (index_type == phi::DataType::INT32) {
    GraphSendUERecvGradOpCUDAKernelLaunchHelper<Context, T, int32_t>(
        ctx,
        out_grad,
        x,
        y,
        src_index,
        dst_index,
        message_op,
        reduce_op,
        x_grad,
        y_grad,
        dst_count.get_ptr(),
        out.get_ptr());
  } else if (index_type == phi::DataType::INT64) {
    GraphSendUERecvGradOpCUDAKernelLaunchHelper<Context, T, int64_t>(
        ctx,
        out_grad,
        x,
        y,
        src_index,
        dst_index,
        message_op,
        reduce_op,
        x_grad,
        y_grad,
        dst_count.get_ptr(),
        out.get_ptr());
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(send_ue_recv_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SendUERecvGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
