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

#include <thrust/device_vector.h>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/gpu/graph_send_ue_recv_funcs.h"
#include "paddle/phi/kernels/impl/graph_message_passing_impl.h"

namespace phi {

template <typename T, typename IndexT, typename ComputeFunctor>
__global__ void GraphSendUVCUDAKernel(const T* x_data,
                                      const T* y_data,
                                      const IndexT* src_indices,
                                      const IndexT* dst_indices,
                                      const int64_t* xbcast_off,
                                      const int64_t* ybcast_off,
                                      T* output,
                                      int64_t index_size,
                                      int64_t x_len,
                                      int64_t y_len,
                                      int64_t out_len,
                                      bool use_bcast,
                                      ComputeFunctor cfunctor) {
  IndexT ty = blockIdx.y * blockDim.y + threadIdx.y;
  const IndexT stride_y = blockDim.y * gridDim.y;

  while (ty < index_size) {
    IndexT src = src_indices[ty];
    IndexT dst = dst_indices[ty];
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_x = blockDim.x * gridDim.x;

    const T* x_off = x_data + src * x_len;
    const T* y_off = y_data + dst * y_len;
    T* out_off = output + ty * out_len;
    while (tx < out_len) {
      int64_t x_add = use_bcast ? xbcast_off[tx] : tx;
      int64_t y_add = use_bcast ? ybcast_off[tx] : tx;
      T val = cfunctor(x_off[x_add], y_off[y_add]);
      out_off[tx] = val;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename Context, typename T, typename IndexT>
void GraphSendUVOpCUDAKernelLaunchHelper(const Context& ctx,
                                         const DenseTensor& x,
                                         const DenseTensor& y,
                                         const DenseTensor& src_index,
                                         const DenseTensor& dst_index,
                                         const std::string& message_op,
                                         DenseTensor* out) {
  const int64_t& index_size = src_index.dims()[0];
  PADDLE_ENFORCE_GT(
      index_size,
      0,
      errors::InvalidArgument("The first dimension of src_index or dst_index "
                              "shoule be greater than 0, but received %d.",
                              index_size));

  auto out_dims = out->dims();
  int64_t memset_size = 1;
  for (int i = 0; i < out_dims.size(); i++) {
    memset_size *= out_dims[i];
  }
  ctx.template Alloc<T>(out);
  T* out_data = out->data<T>();

  const auto& bcast_info = phi::CalcBCastInfo(x.dims(), y.dims());
  const T* x_data = x.data<T>();
  const T* y_data = y.data<T>();
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();

  thrust::device_vector<int64_t> x_bcastoff, y_bcastoff;
  if (bcast_info.use_bcast) {
    CopyBCastOff(bcast_info, &x_bcastoff, &y_bcastoff);
  }

  int64_t out_len = bcast_info.out_len;
  const int ntx = FindNumThreads(out_len, ctx.GetMaxThreadsPerBlock());
  const int nty = ctx.GetMaxThreadsPerBlock() / ntx;
  const int nbx = (out_len + ntx - 1) / ntx;
  const int nby = FindNumBlocks('y', (index_size + nty - 1) / nty);
  const dim3 grid(nbx, nby);
  const dim3 block(ntx, nty);
  if (message_op == "ADD") {
    funcs::AddFunctor<T> add_functor;
    GraphSendUVCUDAKernel<T, IndexT, funcs::AddFunctor<T>>
        <<<grid, block, 0, ctx.stream()>>>(
            x_data,
            y_data,
            s_index,
            d_index,
            thrust::raw_pointer_cast(x_bcastoff.data()),
            thrust::raw_pointer_cast(y_bcastoff.data()),
            out_data,
            index_size,
            bcast_info.l_len,
            bcast_info.r_len,
            out_len,
            bcast_info.use_bcast,
            add_functor);
  } else if (message_op == "MUL") {
    funcs::MultiplyFunctor<T> mul_functor;
    GraphSendUVCUDAKernel<T, IndexT, funcs::MultiplyFunctor<T>>
        <<<grid, block, 0, ctx.stream()>>>(
            x_data,
            y_data,
            s_index,
            d_index,
            thrust::raw_pointer_cast(x_bcastoff.data()),
            thrust::raw_pointer_cast(y_bcastoff.data()),
            out_data,
            index_size,
            bcast_info.l_len,
            bcast_info.r_len,
            out_len,
            bcast_info.use_bcast,
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
    GraphSendUVOpCUDAKernelLaunchHelper<Context, T, int32_t>(
        ctx, x, y, src_index, dst_index, message_op, out);
  } else if (index_type == phi::DataType::INT64) {
    GraphSendUVOpCUDAKernelLaunchHelper<Context, T, int64_t>(
        ctx, x, y, src_index, dst_index, message_op, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(send_uv,
                   GPU,
                   ALL_LAYOUT,
                   phi::SendUVKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
