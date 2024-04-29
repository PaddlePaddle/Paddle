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

#include <algorithm>
#include <set>
#include <vector>

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/graph_send_ue_recv_funcs.h"
#include "paddle/phi/kernels/impl/graph_message_passing_impl.h"

namespace phi {

template <typename T, typename IndexT, typename ComputeFunctor>
void GraphSendUERecvSumCpuKernel(const BroadCastInfo& bcast,
                                 const T* x_data,
                                 const T* y_data,
                                 const IndexT* src_indices,
                                 const IndexT* dst_indices,
                                 T* output,
                                 int64_t index_size,
                                 ComputeFunctor cfunctor) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < index_size; i++) {
    IndexT src = src_indices[i];
    IndexT dst = dst_indices[i];
    T* out_off = output + dst * bcast.out_len;
    const T* x_off = x_data + src * bcast.l_len;
    const T* y_off = y_data + i * bcast.r_len;
    for (int64_t j = 0; j < bcast.out_len; j++) {
      int64_t x_add = bcast.use_bcast ? bcast.l_offset[j] : j;
      int64_t y_add = bcast.use_bcast ? bcast.r_offset[j] : j;
      T val = cfunctor(x_off[x_add], y_off[y_add]);
      if (val != 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp atomic
#endif
        out_off[j] += val;
      }
    }
  }
}

template <typename T,
          typename IndexT,
          typename ComputeFunctor,
          typename CmpFunctor>
void GraphSendUERecvMinMaxCpuKernel(const BroadCastInfo& bcast,
                                    const T* x_data,
                                    const T* y_data,
                                    const IndexT* src_indices,
                                    const IndexT* dst_indices,
                                    T* output,
                                    int64_t index_size,
                                    ComputeFunctor cfunctor,
                                    CmpFunctor pfunctor) {
  std::set<IndexT> existed_dst;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < index_size; i++) {
    IndexT src = src_indices[i];
    IndexT dst = dst_indices[i];
    T* out_off = output + dst * bcast.out_len;
    const T* x_off = x_data + src * bcast.l_len;
    const T* y_off = y_data + i * bcast.r_len;
    bool in_set = existed_dst.find(dst) != existed_dst.end();
    for (int64_t j = 0; j < bcast.out_len; j++) {
      int64_t x_add = bcast.use_bcast ? bcast.l_offset[j] : j;
      int64_t y_add = bcast.use_bcast ? bcast.r_offset[j] : j;
      T val = cfunctor(x_off[x_add], y_off[y_add]);
#ifdef PADDLE_WITH_MKLML
#pragma omp critical
#endif
      if (!in_set) {
        out_off[j] = val;
      } else {
        out_off[j] = pfunctor(out_off[j], val);
      }
    }
#ifdef PADDLE_WITH_MKLML
#pragma omp critical
#endif
    if (!in_set) {
      existed_dst.emplace(dst);
    }
  }
}

template <typename Context, typename T, typename IndexT>
void GraphSendUERecvOpKernelLaunchHelper(const Context& ctx,
                                         const DenseTensor& x,
                                         const DenseTensor& y,
                                         const DenseTensor& src_index,
                                         const DenseTensor& dst_index,
                                         const std::string& message_op,
                                         const std::string& reduce_op,
                                         int64_t out_size,
                                         DenseTensor* out,
                                         DenseTensor* dst_count = nullptr) {
  const int& index_size = src_index.dims()[0];  // NOLINT
  auto out_dims = out->dims();
  int64_t memset_size = 1;
  std::vector<int64_t> dims_ = common::vectorize(out_dims);
  if (out_size <= 0) {
    dims_[0] = x.dims()[0];
  } else {
    dims_[0] = out_size;
  }
  out->Resize(common::make_ddim(dims_));
  for (auto dim : dims_) {
    memset_size *= dim;
  }

  ctx.template Alloc<T>(out);
  T* out_data = out->data<T>();
  const size_t& memset_bytes = memset_size * sizeof(T);
  memset(out_data, 0, memset_bytes);

  if (index_size == 0) return;
  const auto& bcast_info = phi::CalcBCastInfo(x.dims(), y.dims());
  const T* x_data = x.data<T>();
  const T* y_data = y.data<T>();
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();
  if (reduce_op == "SUM" || reduce_op == "MEAN") {
    if (message_op == "ADD") {
      GraphAddFunctor<T> add_functor;
      GraphSendUERecvSumCpuKernel<T, IndexT, GraphAddFunctor<T>>(bcast_info,
                                                                 x_data,
                                                                 y_data,
                                                                 s_index,
                                                                 d_index,
                                                                 out_data,
                                                                 index_size,
                                                                 add_functor);
    } else if (message_op == "MUL") {
      GraphMulFunctor<T> mul_functor;
      GraphSendUERecvSumCpuKernel<T, IndexT, GraphMulFunctor<T>>(bcast_info,
                                                                 x_data,
                                                                 y_data,
                                                                 s_index,
                                                                 d_index,
                                                                 out_data,
                                                                 index_size,
                                                                 mul_functor);
    }
    if (reduce_op == "MEAN") {
      int64_t input_size = out_size <= 0 ? x.dims()[0] : out_size;
      dst_count->Resize({input_size});
      int* dst_count_data = ctx.template Alloc<int>(dst_count);
      memset(dst_count_data, 0, input_size * sizeof(int));
      for (int i = 0; i < index_size; i++) {
        IndexT dst_idx = d_index[i];
        dst_count_data[dst_idx] += 1;
      }
      for (int i = 0; i < input_size; i++) {
        if (dst_count_data[i] == 0) continue;
        auto out_slice = out->Slice(i, i + 1);
        auto eigen_out = phi::EigenVector<T>::Flatten(out_slice);
        eigen_out = eigen_out / static_cast<T>(dst_count_data[i]);
      }
    }
  } else if (reduce_op == "MIN") {
    GraphMinFunctor<T> min_functor;
    if (message_op == "ADD") {
      GraphAddFunctor<T> add_functor;
      GraphSendUERecvMinMaxCpuKernel<T,
                                     IndexT,
                                     GraphAddFunctor<T>,
                                     GraphMinFunctor<T>>(bcast_info,
                                                         x_data,
                                                         y_data,
                                                         s_index,
                                                         d_index,
                                                         out_data,
                                                         index_size,
                                                         add_functor,
                                                         min_functor);
    } else if (message_op == "MUL") {
      GraphMulFunctor<T> mul_functor;
      GraphSendUERecvMinMaxCpuKernel<T,
                                     IndexT,
                                     GraphMulFunctor<T>,
                                     GraphMinFunctor<T>>(bcast_info,
                                                         x_data,
                                                         y_data,
                                                         s_index,
                                                         d_index,
                                                         out_data,
                                                         index_size,
                                                         mul_functor,
                                                         min_functor);
    }
  } else if (reduce_op == "MAX") {
    GraphMaxFunctor<T> max_functor;
    if (message_op == "ADD") {
      GraphAddFunctor<T> add_functor;
      GraphSendUERecvMinMaxCpuKernel<T,
                                     IndexT,
                                     GraphAddFunctor<T>,
                                     GraphMaxFunctor<T>>(bcast_info,
                                                         x_data,
                                                         y_data,
                                                         s_index,
                                                         d_index,
                                                         out_data,
                                                         index_size,
                                                         add_functor,
                                                         max_functor);
    } else if (message_op == "MUL") {
      GraphMulFunctor<T> mul_functor;
      GraphSendUERecvMinMaxCpuKernel<T,
                                     IndexT,
                                     GraphMulFunctor<T>,
                                     GraphMaxFunctor<T>>(bcast_info,
                                                         x_data,
                                                         y_data,
                                                         s_index,
                                                         d_index,
                                                         out_data,
                                                         index_size,
                                                         mul_functor,
                                                         max_functor);
    }
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
    GraphSendUERecvOpKernelLaunchHelper<Context, T, int32_t>(ctx,
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
    GraphSendUERecvOpKernelLaunchHelper<Context, T, int64_t>(ctx,
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
                   CPU,
                   ALL_LAYOUT,
                   phi::SendUERecvKernel,
                   float,
                   double,
                   int,
                   int64_t) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
}
