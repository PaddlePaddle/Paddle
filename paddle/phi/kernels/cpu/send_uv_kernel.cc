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

#include "paddle/phi/kernels/send_uv_kernel.h"

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/graph_send_ue_recv_funcs.h"
#include "paddle/phi/kernels/impl/graph_message_passing_impl.h"

namespace phi {

template <typename T, typename IndexT, typename ComputeFunctor>
void GraphSendUVCpuKernel(const BroadCastInfo& bcast,
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
    T* out_off = output + i * bcast.out_len;
    const T* x_off = x_data + src * bcast.l_len;
    const T* y_off = y_data + dst * bcast.r_len;
    for (int64_t j = 0; j < bcast.out_len; j++) {
      int64_t x_add = bcast.use_bcast ? bcast.l_offset[j] : j;
      int64_t y_add = bcast.use_bcast ? bcast.r_offset[j] : j;
      T val = cfunctor(x_off[x_add], y_off[y_add]);
      out_off[j] = val;
    }
  }
}

template <typename Context, typename T, typename IndexT>
void GraphSendUVOpKernelLaunchHelper(const Context& ctx,
                                     const DenseTensor& x,
                                     const DenseTensor& y,
                                     const DenseTensor& src_index,
                                     const DenseTensor& dst_index,
                                     const std::string& message_op,
                                     DenseTensor* out) {
  const int& index_size = src_index.dims()[0];  // NOLINT
  PADDLE_ENFORCE_GT(
      index_size,
      0,
      errors::InvalidArgument("The first dimension of src_index or dst_index "
                              "should be greater than 0, but received %d.",
                              index_size));

  ctx.template Alloc<T>(out);
  T* out_data = out->data<T>();

  const auto& bcast_info = phi::CalcBCastInfo(x.dims(), y.dims());
  const T* x_data = x.data<T>();
  const T* y_data = y.data<T>();
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();
  if (message_op == "ADD") {
    GraphAddFunctor<T> add_functor;
    GraphSendUVCpuKernel<T, IndexT, GraphAddFunctor<T>>(bcast_info,
                                                        x_data,
                                                        y_data,
                                                        s_index,
                                                        d_index,
                                                        out_data,
                                                        index_size,
                                                        add_functor);
  } else if (message_op == "MUL") {
    GraphMulFunctor<T> mul_functor;
    GraphSendUVCpuKernel<T, IndexT, GraphMulFunctor<T>>(bcast_info,
                                                        x_data,
                                                        y_data,
                                                        s_index,
                                                        d_index,
                                                        out_data,
                                                        index_size,
                                                        mul_functor);
  }
}

template <typename T, typename Context>
void SendUVKernel(const Context& ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  const DenseTensor& src_index,
                  const DenseTensor& dst_index,
                  const std::string& message_op,
                  DenseTensor* out) {
  auto index_type = src_index.dtype();
  if (index_type == phi::DataType::INT32) {
    GraphSendUVOpKernelLaunchHelper<Context, T, int32_t>(
        ctx, x, y, src_index, dst_index, message_op, out);
  } else if (index_type == phi::DataType::INT64) {
    GraphSendUVOpKernelLaunchHelper<Context, T, int64_t>(
        ctx, x, y, src_index, dst_index, message_op, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    send_uv, CPU, ALL_LAYOUT, phi::SendUVKernel, float, double, int, int64_t) {}
