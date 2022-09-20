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

#include "paddle/phi/kernels/graph_send_ue_recv_grad_kernel.h"

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/graph_send_recv_funcs.h"
#include "paddle/phi/kernels/cpu/graph_send_ue_recv_funcs.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/graph_message_passing_impl.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

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
                    T* x_grad,
                    const DenseTensor& out_grad_tensor,
                    DenseTensor* x_grad_tensor,
                    const DenseTensor* dst_count = nullptr,
                    const DenseTensor* out = nullptr) {
  std::vector<int64_t> reduce_idx;
  bool reduce = ReduceGrad(out_grad_dims, x_dims, reduce_idx);

  if (reduce_op == "SUM") {
    if (message_op == "ADD") {
      GraphSendRecvSumFunctor<T> sum_functor;
      if (!reduce) {
        for (int64_t i = 0; i < index_size; i++) {
          IndexT src = s_index[i];
          IndexT dst = d_index[i];
          ElementwiseInnerOperation<T, IndexT, GraphSendRecvSumFunctor<T>>(
              out_grad_tensor, x_grad_tensor, src, dst, false, sum_functor);
        }
      } else {
        DenseTensor x_grad_v2 =
            phi::EmptyLike<T, Context>(ctx, out_grad_tensor);
        phi::funcs::SetConstant<Context, T>()(ctx, &x_grad_v2, T(0));
        for (int64_t i = 0; i < index_size; i++) {
          IndexT src = s_index[i];
          IndexT dst = d_index[i];
          ElementwiseInnerOperation<T, IndexT, GraphSendRecvSumFunctor<T>>(
              out_grad_tensor, &x_grad_v2, src, dst, false, sum_functor);
        }
        DenseTensor x_grad_out = phi::Sum<T, Context>(
            ctx,
            x_grad_v2,
            phi::IntArray(reduce_idx),
            paddle::experimental::CppTypeToDataType<T>::Type(),
            true);
        memcpy(x_grad, x_grad_out.data<T>(), x_grad_out.numel() * sizeof(T));
      }
    } else if (message_op == "MUL") {
      const auto& bcast = phi::CalcBCastInfo(out_grad_dims, e_dims);
      if (!reduce) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int64_t i = 0; i < index_size; i++) {
          IndexT src = s_index[i];
          IndexT dst = d_index[i];
          T* x_grad_off = x_grad + dst * bcast.out_len;
          const T* out_grad_off = out_grad + src * bcast.l_len;
          const T* e_off = e_data + i * bcast.r_len;
          for (int j = 0; j < bcast.out_len; j++) {
            int64_t o_add = bcast.use_bcast ? bcast.l_offset[j] : j;
            int64_t e_add = bcast.use_bcast ? bcast.r_offset[j] : j;
            T val = out_grad_off[o_add] * e_off[e_add];
            if (val != 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
              x_grad_off[j] += val;
            }
          }
        }
      } else {
        DenseTensor x_grad_v2 =
            phi::EmptyLike<T, Context>(ctx, out_grad_tensor);
        phi::funcs::SetConstant<Context, T>()(ctx, &x_grad_v2, T(0));
        T* x_grad_v2_data = x_grad_v2.data<T>();
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int64_t i = 0; i < index_size; i++) {
          IndexT src = s_index[i];
          IndexT dst = d_index[i];
          T* x_grad_off = x_grad_v2_data + dst * bcast.out_len;
          const T* out_grad_off = out_grad + src * bcast.l_len;
          const T* e_off = e_data + i * bcast.r_len;
          for (int j = 0; j < bcast.out_len; j++) {
            int64_t o_add = bcast.use_bcast ? bcast.l_offset[j] : j;
            int64_t e_add = bcast.use_bcast ? bcast.r_offset[j] : j;
            T val = out_grad_off[o_add] * e_off[e_add];
            if (val != 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
              x_grad_off[j] += val;
            }
          }
        }
        DenseTensor x_grad_out = phi::Sum<T, Context>(
            ctx,
            x_grad_v2,
            phi::IntArray(reduce_idx),
            paddle::experimental::CppTypeToDataType<T>::Type(),
            true);
        memcpy(x_grad, x_grad_out.data<T>(), x_grad_out.numel() * sizeof(T));
      }
    }
  } else if (reduce_op == "MEAN") {
    const int* s_count = dst_count->data<int>();
    if (message_op == "ADD") {
      if (!reduce) {
        for (int64_t i = 0; i < index_size; i++) {
          IndexT src = s_index[i];
          IndexT dst = d_index[i];
          auto out_grad_slice = out_grad_tensor.Slice(src, src + 1);
          auto x_grad_slice = x_grad_tensor->Slice(dst, dst + 1);
          auto eigen_out_grad = phi::EigenVector<T>::Flatten(out_grad_slice);
          auto eigen_x_grad = phi::EigenVector<T>::Flatten(x_grad_slice);
          eigen_x_grad += (eigen_out_grad / static_cast<T>(s_count[src]));
        }
      } else {
        DenseTensor x_grad_v2 =
            phi::EmptyLike<T, Context>(ctx, out_grad_tensor);
        phi::funcs::SetConstant<Context, T>()(ctx, &x_grad_v2, T(0));
        for (int64_t i = 0; i < index_size; i++) {
          IndexT src = s_index[i];
          IndexT dst = d_index[i];
          auto out_grad_slice = out_grad_tensor.Slice(src, src + 1);
          auto x_grad_slice = x_grad_v2.Slice(dst, dst + 1);
          auto eigen_out_grad = phi::EigenVector<T>::Flatten(out_grad_slice);
          auto eigen_x_grad = phi::EigenVector<T>::Flatten(x_grad_slice);
          eigen_x_grad += (eigen_out_grad / static_cast<T>(s_count[src]));
        }
        DenseTensor x_grad_out = phi::Sum<T, Context>(
            ctx,
            x_grad_v2,
            phi::IntArray(reduce_idx),
            paddle::experimental::CppTypeToDataType<T>::Type(),
            true);
        memcpy(x_grad, x_grad_out.data<T>(), x_grad_out.numel() * sizeof(T));
      }
    } else if (message_op == "MUL") {
      const auto& bcast = phi::CalcBCastInfo(out_grad_dims, e_dims);
      if (!reduce) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int64_t i = 0; i < index_size; i++) {
          IndexT src = s_index[i];
          IndexT dst = d_index[i];
          const T* out_grad_off = out_grad + src * bcast.l_len;
          const T* e_off = e_data + i * bcast.r_len;
          T* x_grad_off = x_grad + dst * bcast.out_len;
          for (int64_t j = 0; j < bcast.out_len; j++) {
            int64_t o_add = bcast.use_bcast ? bcast.l_offset[j] : j;
            int64_t e_add = bcast.use_bcast ? bcast.r_offset[j] : j;
            T val = out_grad_off[o_add] * e_off[e_add];
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
            x_grad_off[j] += (val / s_count[src]);
          }
        }
      } else {
        DenseTensor x_grad_v2 =
            phi::EmptyLike<T, Context>(ctx, out_grad_tensor);
        phi::funcs::SetConstant<Context, T>()(ctx, &x_grad_v2, T(0));
        T* x_grad_v2_data = x_grad_v2.data<T>();
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int64_t i = 0; i < index_size; i++) {
          IndexT src = s_index[i];
          IndexT dst = d_index[i];
          const T* out_grad_off = out_grad + src * bcast.l_len;
          const T* e_off = e_data + i * bcast.r_len;
          T* x_grad_off = x_grad_v2_data + dst * bcast.out_len;
          for (int64_t j = 0; j < bcast.out_len; j++) {
            int64_t o_add = bcast.use_bcast ? bcast.l_offset[j] : j;
            int64_t e_add = bcast.use_bcast ? bcast.r_offset[j] : j;
            T val = out_grad_off[o_add] * e_off[e_add];
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
            x_grad_off[j] += (val / s_count[src]);
          }
        }
        DenseTensor x_grad_out = phi::Sum<T, Context>(
            ctx,
            x_grad_v2,
            phi::IntArray(reduce_idx),
            paddle::experimental::CppTypeToDataType<T>::Type(),
            true);
        memcpy(x_grad, x_grad_out.data<T>(), x_grad_out.numel() * sizeof(T));
      }
    }
  }
}

template <typename T, typename IndexT>
void CalculateEGrad(const T* out_grad_data,
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
  const auto& bcast = phi::CalcBCastInfo(x_dims, e_dims);
  if (reduce_op == "SUM") {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < index_size; i++) {
      IndexT src = s_index[i];
      IndexT dst = d_index[i];
      const T* x_off = x_data + src * bcast.l_len;
      const T* out_grad_off = out_grad_data + dst * bcast.out_len;
      T* e_grad_off = e_grad + i * bcast.r_len;
      for (int64_t j = 0; j < bcast.out_len; j++) {
        int64_t x_add = bcast.use_bcast ? bcast.l_offset[j] : j;
        int64_t e_add = bcast.use_bcast ? bcast.r_offset[j] : j;
        if (message_op == "ADD") {
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
          e_grad_off[e_add] += out_grad_off[j];
        } else if (message_op == "MUL") {
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
          e_grad_off[e_add] += (out_grad_off[j] * x_off[x_add]);
        }
      }
    }
  } else if (reduce_op == "MEAN") {
    const int* s_count = dst_count->data<int>();
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < index_size; i++) {
      IndexT src = s_index[i];
      IndexT dst = d_index[i];
      const T* x_off = x_data + src * bcast.l_len;
      const T* out_grad_off = out_grad_data + dst * bcast.out_len;
      T* e_grad_off = e_grad + i * bcast.r_len;
      for (int64_t j = 0; j < bcast.out_len; j++) {
        int64_t x_add = bcast.use_bcast ? bcast.l_offset[j] : j;
        int64_t e_add = bcast.use_bcast ? bcast.r_offset[j] : j;
        if (message_op == "ADD") {
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
          e_grad_off[e_add] += (out_grad_off[j] / s_count[dst]);
        } else if (message_op == "MUL") {
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
          e_grad_off[e_add] += (out_grad_off[j] * x_off[x_add] / s_count[dst]);
        }
      }
    }
  }
}

template <typename T, typename IndexT>
void CalculateXEGradForMinMax(const T* out_grad,
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
  const auto& bcast = phi::CalcBCastInfo(x_dims, e_dims);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < index_size; i++) {
    IndexT src = s_index[i];
    IndexT dst = d_index[i];
    const T* x_off = x_data + dst * bcast.l_len;
    const T* e_off = e_data + i * bcast.r_len;
    const T* out_off = out_data + src * bcast.out_len;
    const T* out_grad_off = out_grad + src * bcast.out_len;
    T* x_grad_off = x_grad + dst * bcast.l_len;
    T* e_grad_off = e_grad + i * bcast.r_len;
    for (int64_t j = 0; j < bcast.out_len; j++) {
      int64_t x_add = bcast.use_bcast ? bcast.l_offset[j] : j;
      int64_t e_add = bcast.use_bcast ? bcast.r_offset[j] : j;
      if (message_op == "ADD") {
        T val = x_off[x_add] + e_off[e_add];
#ifdef PADDLE_WITH_MKLML
#pragma omp critical
#endif
        x_grad_off[x_add] += (out_grad_off[j] * (val == out_off[j]));
        e_grad_off[e_add] += (out_grad_off[j] * (val == out_off[j]));
      } else if (message_op == "MUL") {
        T val = x_off[x_add] * e_off[e_add];
#ifdef PADDLE_WITH_MKLML
#pragma omp critical
#endif
        x_grad_off[x_add] +=
            (out_grad_off[j] * (val == out_off[j]) * e_off[e_add]);
        e_grad_off[e_add] +=
            (out_grad_off[j] * (val == out_off[j]) * x_off[x_add]);
      }
    }
  }
}

template <typename Context, typename T, typename IndexT>
void GraphSendUERecvGradOpKernelLaunchHelper(
    const Context& ctx,
    const DenseTensor& out_grad,
    const DenseTensor& x,
    const DenseTensor& y,
    const DenseTensor& src_index,
    const DenseTensor& dst_index,
    const std::string& message_op,
    const std::string& reduce_op,
    DenseTensor* x_grad,
    DenseTensor* y_grad,
    const DenseTensor* dst_count = nullptr,
    const DenseTensor* out = nullptr) {
  const int& index_size = dst_index.dims()[0];

  ctx.template Alloc<T>(x_grad);
  T* x_grad_data = x_grad->data<T>();
  ctx.template Alloc<T>(y_grad);
  T* y_grad_data = y_grad->data<T>();
  const auto& x_dims = x.dims();
  const auto& y_dims = y.dims();
  int64_t memset_size_x = 1, memset_size_y = 1;
  int64_t slice_size = 1;
  for (int i = 0; i < x_dims.size(); i++) {
    memset_size_x *= x_dims[i];
    if (i > 0) slice_size *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); i++) {
    memset_size_y *= y_dims[i];
  }
  const size_t& memset_bytes_x = memset_size_x * sizeof(T);
  const size_t& memset_bytes_y = memset_size_y * sizeof(T);
  memset(x_grad_data, 0, memset_bytes_x);
  memset(y_grad_data, 0, memset_bytes_y);

  if (index_size == 0) return;

  const T* out_grad_data = out_grad.data<T>();
  const T* x_data = x.data<T>();
  const T* y_data = y.data<T>();
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();

  if (reduce_op == "SUM" || reduce_op == "MEAN") {
    CalculateXGrad<Context, T, IndexT>(ctx,
                                       out_grad_data,
                                       x_data,
                                       y_data,
                                       out_grad.dims(),
                                       x_dims,
                                       y_dims,
                                       d_index,
                                       s_index,
                                       message_op,
                                       reduce_op,
                                       index_size,
                                       x_grad_data,
                                       out_grad,
                                       x_grad,
                                       dst_count,
                                       out);
    CalculateEGrad<T, IndexT>(out_grad_data,
                              x_data,
                              y_data,
                              x_dims,
                              y_dims,
                              s_index,
                              d_index,
                              message_op,
                              reduce_op,
                              index_size,
                              y_grad_data,
                              dst_count);
  } else if (reduce_op == "MIN" || reduce_op == "MAX") {
    CalculateXEGradForMinMax<T, IndexT>(out_grad_data,
                                        x_data,
                                        y_data,
                                        x_dims,
                                        y_dims,
                                        d_index,
                                        s_index,
                                        message_op,
                                        reduce_op,
                                        index_size,
                                        x_grad_data,
                                        y_grad_data,
                                        out);
  }
}

template <typename T, typename Context>
void GraphSendUERecvGradKernel(const Context& ctx,
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
    GraphSendUERecvGradOpKernelLaunchHelper<Context, T, int32_t>(
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
    GraphSendUERecvGradOpKernelLaunchHelper<Context, T, int64_t>(
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

PD_REGISTER_KERNEL(graph_send_ue_recv_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::GraphSendUERecvGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
