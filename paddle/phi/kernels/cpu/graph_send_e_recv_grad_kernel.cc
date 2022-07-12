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

#include "paddle/phi/kernels/graph_send_e_recv_grad_kernel.h"

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/graph_send_e_recv_funcs.h"
#include "paddle/phi/kernels/impl/graph_send_e_recv_kernel_impl.h"

namespace phi {

template <typename T, typename IndexT>
void CalculateXGrad(const T* out_grad,
                    const T* x_data,
                    const T* e_data,
                    const phi::DDim& out_grad_dims,
                    const phi::DDim& x_dims,
                    const phi::DDim& e_dims,
                    const IndexT* s_index,
                    const IndexT* d_index,
                    const std::string& compute_type,
                    const std::string& pool_type,
                    int64_t index_size,
                    int64_t slice_size,
                    T* x_grad,
                    const DenseTensor* dst_count = nullptr,
                    const DenseTensor* out = nullptr) {
  if (pool_type == "SUM") {
    if (compute_type == "ADD") {
    } else if (compute_type == "MUL") {
    }
  } else if (pool_type == "MEAN") {
    if (compute_type == "ADD") {
    } else if (compute_type == "MUL") {
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
                    const std::string& compute_type,
                    const std::string& pool_type,
                    int64_t index_size,
                    T* e_grad,
                    const DenseTensor* dst_count = nullptr) {
  const auto& bcast = phi::CalcBCastInfo(x_dims, e_dims);
  if (pool_type == "SUM") {
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
        if (compute_type == "ADD") {
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
          e_grad_off[e_add] += out_grad_off[j];
        } else if (compute_type == "MUL") {
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
          e_grad_off[e_add] += (out_grad_off[j] * x_off[x_add]);
        }
      }
    }
  } else if (pool_type == "MEAN") {
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
        if (compute_type == "ADD") {
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
          e_grad_off[e_add] += (out_grad_off[j] / s_count[dst]);
        } else if (compute_type == "MUL") {
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
                              const std::string& compute_type,
                              const std::string& pool_type,
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
      if (compute_type == "ADD") {
        T val = x_off[x_add] + e_off[e_add];
#ifdef PADDLE_WITH_MKLML
#pragma omp critical
#endif
        x_grad_off[x_add] += (out_grad_off[j] * (val == out_off[j]));
        e_grad_off[e_add] += (out_grad_off[j] * (val == out_off[j]));
      } else if (compute_type == "MUL") {
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
void GraphSendERecvGradOpKernelLaunchHelper(
    const Context& ctx,
    const DenseTensor& out_grad,
    const DenseTensor& x,
    const DenseTensor& e,
    const DenseTensor& src_index,
    const DenseTensor& dst_index,
    const std::string& compute_type,
    const std::string& pool_type,
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
  memset(x_grad_data, 0, memset_bytes_x);
  memset(e_grad_data, 0, memset_bytes_e);

  if (index_size == 0) return;

  const T* out_grad_data = out_grad.data<T>();
  const T* x_data = x.data<T>();
  const T* e_data = e.data<T>();
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();

  if (pool_type == "SUM" || pool_type == "MEAN") {
    CalculateEGrad<T, IndexT>(out_grad_data,
                              x_data,
                              e_data,
                              x_dims,
                              e_dims,
                              s_index,
                              d_index,
                              compute_type,
                              pool_type,
                              index_size,
                              e_grad_data,
                              dst_count);
  } else if (pool_type == "MIN" || pool_type == "MAX") {
    CalculateXEGradForMinMax<T, IndexT>(out_grad_data,
                                        x_data,
                                        e_data,
                                        x_dims,
                                        e_dims,
                                        d_index,
                                        s_index,
                                        compute_type,
                                        pool_type,
                                        index_size,
                                        x_grad_data,
                                        e_grad_data,
                                        out);
  }
}

template <typename T, typename Context>
void GraphSendERecvGradKernel(const Context& ctx,
                              const DenseTensor& x,
                              const DenseTensor& e,
                              const DenseTensor& src_index,
                              const DenseTensor& dst_index,
                              const paddle::optional<DenseTensor>& out,
                              const paddle::optional<DenseTensor>& dst_count,
                              const DenseTensor& out_grad,
                              const std::string& compute_type,
                              const std::string& pool_type,
                              DenseTensor* x_grad,
                              DenseTensor* e_grad) {
  auto index_type = src_index.dtype();
  if (index_type == phi::DataType::INT32) {
    GraphSendERecvGradOpKernelLaunchHelper<Context, T, int32_t>(
        ctx,
        out_grad,
        x,
        e,
        src_index,
        dst_index,
        compute_type,
        pool_type,
        x_grad,
        e_grad,
        dst_count.get_ptr(),
        out.get_ptr());
  } else if (index_type == phi::DataType::INT64) {
    GraphSendERecvGradOpKernelLaunchHelper<Context, T, int64_t>(
        ctx,
        out_grad,
        x,
        e,
        src_index,
        dst_index,
        compute_type,
        pool_type,
        x_grad,
        e_grad,
        dst_count.get_ptr(),
        out.get_ptr());
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(graph_send_e_recv_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::GraphSendERecvGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
