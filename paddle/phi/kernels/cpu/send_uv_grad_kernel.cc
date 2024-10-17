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

#include "paddle/phi/kernels/send_uv_grad_kernel.h"

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/graph_message_passing_impl.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename Context, typename T, typename IndexT>
void CalculateGrad(const Context& ctx,
                   const T* out_grad,
                   const IndexT* s_index,
                   const IndexT* d_index,
                   const phi::DDim& out_grad_dims,
                   const phi::DDim& x_grad_dims,
                   const std::string& message_op,
                   int64_t index_size,
                   int64_t slice_size,
                   T* x_grad,
                   const DenseTensor& out_grad_tensor UNUSED,
                   const DenseTensor& y) {
  std::vector<int64_t> reduce_idx;
  bool reduce = ReduceGrad(out_grad_dims, x_grad_dims, reduce_idx);

  if (message_op == "ADD") {
    if (!reduce) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int64_t i = 0; i < index_size; i++) {
        IndexT dst = d_index[i];
        T* x_grad_off = x_grad + dst * slice_size;
        const T* out_grad_off = out_grad + i * slice_size;
        for (int64_t j = 0; j < slice_size; j++) {
          if (out_grad_off[j] != 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
            x_grad_off[j] += out_grad_off[j];
          }
        }
      }
    } else {
      const auto& bcast_info = phi::CalcBCastInfo(out_grad_dims, x_grad_dims);
      auto out_grad_dims_1 = common::vectorize<int>(out_grad_dims);
      std::vector<int> out_grad_dims_2(out_grad_dims_1.begin() + 1,
                                       out_grad_dims_1.end());
      out_grad_dims_2.emplace(out_grad_dims_2.begin(), x_grad_dims[0]);
      DenseTensor x_grad_v2 = phi::Empty<T, Context>(ctx, out_grad_dims_2);
      phi::funcs::SetConstant<Context, T>()(ctx, &x_grad_v2, static_cast<T>(0));
      T* x_grad_v2_data = x_grad_v2.data<T>();
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int64_t i = 0; i < index_size; i++) {
        IndexT dst = d_index[i];
        T* x_grad_off = x_grad_v2_data + dst * bcast_info.out_len;
        const T* out_grad_off = out_grad + i * bcast_info.out_len;
        for (int64_t j = 0; j < bcast_info.out_len; j++) {
          if (out_grad_off[j] != 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
            x_grad_off[j] += out_grad_off[j];
          }
        }
      }
      DenseTensor x_grad_out =
          phi::Sum<T, Context>(ctx,
                               x_grad_v2,
                               phi::IntArray(reduce_idx),
                               phi::CppTypeToDataType<T>::Type(),
                               true);
      memcpy(x_grad, x_grad_out.data<T>(), x_grad_out.numel() * sizeof(T));
    }
  } else if (message_op == "MUL") {
    const auto& bcast = phi::CalcBCastInfo(y.dims(), out_grad_dims);
    const T* y_data = y.data<T>();
    if (!reduce) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int64_t i = 0; i < index_size; i++) {
        IndexT src = s_index[i];
        IndexT dst = d_index[i];
        T* x_grad_off = x_grad + dst * bcast.out_len;
        const T* y_off = y_data + src * bcast.l_len;
        const T* out_grad_off = out_grad + i * bcast.r_len;
        for (int64_t j = 0; j < bcast.out_len; j++) {
          int64_t y_add = bcast.use_bcast ? bcast.l_offset[j] : j;
          int64_t o_add = bcast.use_bcast ? bcast.r_offset[j] : j;
          T val = y_off[y_add] * out_grad_off[o_add];
          if (val != 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
            x_grad_off[j] += val;
          }
        }
      }
    } else {
      auto out_grad_dims_1 = common::vectorize<int>(out_grad_dims);
      std::vector<int> out_grad_dims_2(out_grad_dims_1.begin() + 1,
                                       out_grad_dims_1.end());
      out_grad_dims_2.emplace(out_grad_dims_2.begin(), x_grad_dims[0]);
      DenseTensor x_grad_v2 = phi::Empty<T, Context>(ctx, out_grad_dims_2);
      phi::funcs::SetConstant<Context, T>()(ctx, &x_grad_v2, static_cast<T>(0));
      T* x_grad_v2_data = x_grad_v2.data<T>();
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int64_t i = 0; i < index_size; i++) {
        IndexT src = s_index[i];
        IndexT dst = d_index[i];
        T* x_grad_off = x_grad_v2_data + dst * bcast.out_len;
        const T* y_off = y_data + src * bcast.l_len;
        const T* out_grad_off = out_grad + i * bcast.r_len;
        for (int64_t j = 0; j < bcast.out_len; j++) {
          int64_t y_add = bcast.use_bcast ? bcast.l_offset[j] : j;
          int64_t o_add = bcast.use_bcast ? bcast.r_offset[j] : j;
          T val = y_off[y_add] * out_grad_off[o_add];
          if (val != 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
            x_grad_off[j] += val;
          }
        }
      }
      DenseTensor x_grad_out =
          phi::Sum<T, Context>(ctx,
                               x_grad_v2,
                               phi::IntArray(reduce_idx),
                               phi::CppTypeToDataType<T>::Type(),
                               true);
      memcpy(x_grad, x_grad_out.data<T>(), x_grad_out.numel() * sizeof(T));
    }
  }
}

template <typename Context, typename T, typename IndexT>
void GraphSendUVGradOpKernelLaunchHelper(const Context& ctx,
                                         const DenseTensor& x,
                                         const DenseTensor& y,
                                         const DenseTensor& out_grad,
                                         const DenseTensor& src_index,
                                         const DenseTensor& dst_index,
                                         const std::string& message_op,
                                         DenseTensor* x_grad,
                                         DenseTensor* y_grad) {
  const int64_t& index_size = dst_index.dims()[0];

  PADDLE_ENFORCE_GT(
      index_size,
      0,
      errors::InvalidArgument("The first dimension of src_index or dst_index "
                              "should be greater than 0, but received %d.",
                              index_size));

  ctx.template Alloc<T>(x_grad);
  T* x_grad_data = x_grad->data<T>();
  ctx.template Alloc<T>(y_grad);
  T* y_grad_data = y_grad->data<T>();
  const auto& x_grad_dims = x_grad->dims();
  const auto& y_grad_dims = y_grad->dims();
  int64_t memset_size_x = 1, memset_size_y = 1;
  int64_t slice_size_x = 1, slice_size_y = 1;
  for (int i = 0; i < x_grad_dims.size(); i++) {
    memset_size_x *= x_grad_dims[i];
    if (i > 0) slice_size_x *= x_grad_dims[i];
  }
  for (int i = 0; i < y_grad_dims.size(); i++) {
    memset_size_y *= y_grad_dims[i];
    if (i > 0) slice_size_y *= y_grad_dims[i];
  }
  const size_t& memset_bytes_x = memset_size_x * sizeof(T);
  const size_t& memset_bytes_y = memset_size_y * sizeof(T);
  memset(x_grad_data, 0, memset_bytes_x);
  memset(y_grad_data, 0, memset_bytes_y);

  const T* out_grad_data = out_grad.data<T>();
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();
  const auto& out_grad_dims = out_grad.dims();
  // Calculate X Grad.
  CalculateGrad<Context, T, IndexT>(ctx,
                                    out_grad_data,
                                    d_index,
                                    s_index,
                                    out_grad_dims,
                                    x_grad_dims,
                                    message_op,
                                    index_size,
                                    slice_size_x,
                                    x_grad_data,
                                    out_grad,
                                    y);
  // Calculate Y Grad.
  CalculateGrad<Context, T, IndexT>(ctx,
                                    out_grad_data,
                                    s_index,
                                    d_index,
                                    out_grad_dims,
                                    y_grad_dims,
                                    message_op,
                                    index_size,
                                    slice_size_y,
                                    y_grad_data,
                                    out_grad,
                                    x);
}

template <typename T, typename Context>
void SendUVGradKernel(const Context& ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& src_index,
                      const DenseTensor& dst_index,
                      const DenseTensor& out_grad,
                      const std::string& message_op,
                      DenseTensor* x_grad,
                      DenseTensor* y_grad) {
  auto index_type = src_index.dtype();
  if (index_type == phi::DataType::INT32) {
    GraphSendUVGradOpKernelLaunchHelper<Context, T, int32_t>(
        ctx, x, y, out_grad, src_index, dst_index, message_op, x_grad, y_grad);
  } else if (index_type == phi::DataType::INT64) {
    GraphSendUVGradOpKernelLaunchHelper<Context, T, int64_t>(
        ctx, x, y, out_grad, src_index, dst_index, message_op, x_grad, y_grad);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(send_uv_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SendUVGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
