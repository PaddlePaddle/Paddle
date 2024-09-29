// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/partial_sum_kernel.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/partial_sum_kernel_impl.h"

namespace phi {

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

template <class T>
__global__ void SumArrayPartialCUDAKernel(T **in,
                                          T *out,
                                          int64_t lod_length,
                                          size_t in_size,
                                          int64_t start_index,
                                          int64_t length,
                                          int64_t row_length) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < lod_length) {
    T total = static_cast<T>(0);
    int b_id = id / length;
    int b_offset = id % length;

    for (int i = 0; i < in_size; ++i) {
      const T *tmp = in[i];
      if (tmp) {
        total += tmp[start_index + b_id * row_length + b_offset];
      }
    }
    out[id] = total;
    id += blockDim.x * gridDim.x;
  }
}

template <class T>
__global__ void PartialSumGradCUDAKernel(T **res_grad,
                                         const T *out_grad,
                                         int64_t lod_length,
                                         size_t in_size,
                                         int64_t start_index,
                                         int64_t length,
                                         int64_t row_length) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < lod_length) {
    T total = static_cast<T>(0);
    int b_id = id / length;
    int b_offset = id % length;

    for (int i = 0; i < in_size; ++i) {
      T *tmp = res_grad[i];
      tmp[start_index + b_id * row_length + b_offset] = out_grad[i];
    }
    id += blockDim.x * gridDim.x;
  }
}

template <typename T, typename Context>
void PartialSumOpCUDAKernel(const Context &dev_ctx,
                            const std::vector<const DenseTensor *> &x,
                            int start_index,
                            int length,
                            DenseTensor *out) {
  auto ctx = dev_ctx;
  auto in_vars = x;

  PADDLE_ENFORCE_EQ(
      x.size() > 0,
      true,
      common::errors::InvalidArgument("The input should not be null."));

  auto place = dev_ctx.GetPlace();  // GPUPlace only now
  auto batch_size = in_vars[0]->dims()[0];
  if (length == -1) {
    length = in_vars[0]->dims()[1] - start_index;
  }

  constexpr size_t theory_sm_threads = 1024;
  auto stream = dev_ctx.stream();
  auto max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  auto sm_count = max_threads / theory_sm_threads;
  size_t tile_size = 0;
  dim3 grids;
  dim3 blocks;
  auto ComputeKernelParameter = [&](size_t length) {
    if (length >= max_threads)
      tile_size = 1024;
    else if (length < max_threads && length > sm_count * 128)
      tile_size = 512;
    else if (length <= sm_count * 128)
      tile_size = 256;
    grids = dim3(CEIL_DIV(length, tile_size), 1, 1);
    blocks = dim3(tile_size, 1, 1);
  };

  auto lod_length = length * batch_size;
  auto row_length = in_vars[0]->dims()[1];
  auto in_num = in_vars.size();

  std::vector<const T *> in_data;
  for (int i = 0; i < in_num; ++i) {
    in_data.emplace_back(in_vars[i]->data<T>());
  }

  if (!in_data.empty()) {
    auto tmp_in_array = phi::memory_utils::Alloc(
        dev_ctx.GetPlace(),
        in_data.size() * sizeof(T *),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

    phi::memory_utils::Copy(dev_ctx.GetPlace(),
                            tmp_in_array->ptr(),
                            phi::CPUPlace(),
                            reinterpret_cast<void *>(in_data.data()),
                            in_data.size() * sizeof(T *));

    T **in_array_data = reinterpret_cast<T **>(tmp_in_array->ptr());
    ComputeKernelParameter(lod_length);
    SumArrayPartialCUDAKernel<T><<<grids, blocks, 0, stream>>>(in_array_data,
                                                               out->data<T>(),
                                                               lod_length,
                                                               in_data.size(),
                                                               start_index,
                                                               length,
                                                               row_length);
  }
}

template <typename T, typename Context>
void PartialSumGradOpCUDAKernel(const Context &dev_ctx,
                                const std::vector<const DenseTensor *> &x,
                                const DenseTensor out_grad,
                                int start_index,
                                int length,
                                std::vector<DenseTensor *> x_grad) {
  auto ins = x;
  auto outs = x_grad;

  PADDLE_ENFORCE_EQ(
      ins.size() > 0,
      true,
      common::errors::InvalidArgument("The input should not be null."));
  if (length == -1) {
    length = ins[0]->dims()[1] - start_index;
  }

  // initialize
  auto &place = *dev_ctx.eigen_device();
  for (size_t i = 0; i < outs.size(); ++i) {
    dev_ctx.template Alloc<T>(outs[i]);
    auto dxt = phi::EigenVector<T>::Flatten(*outs[i]);
    dxt.device(place) = dxt.constant(static_cast<T>(0));
  }

  auto batch_size = ins[0]->dims()[0];
  if (length == -1) {
    length = ins[0]->dims()[1] - start_index;
  }
  auto lod_length = length * batch_size;
  auto row_length = ins[0]->dims()[1];
  auto out_num = outs.size();

  constexpr size_t theory_sm_threads = 1024;
  auto stream = dev_ctx.stream();
  auto max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  auto sm_count = max_threads / theory_sm_threads;
  size_t tile_size = 0;
  dim3 grids;
  dim3 blocks;
  auto ComputeKernelParameter = [&](size_t length) {
    if (length >= max_threads)
      tile_size = 1024;
    else if (length < max_threads && length > sm_count * 128)
      tile_size = 512;
    else if (length <= sm_count * 128)
      tile_size = 256;
    grids = dim3(CEIL_DIV(length, tile_size), 1, 1);
    blocks = dim3(tile_size, 1, 1);
  };

  std::vector<const T *> out_data;
  for (int i = 0; i < out_num; ++i) {
    out_data.emplace_back(outs[i]->data<T>());
  }

  if (!out_data.empty()) {
    auto tmp_out_array = phi::memory_utils::Alloc(
        dev_ctx.GetPlace(),
        out_data.size() * sizeof(T *),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

    phi::memory_utils::Copy(dev_ctx.GetPlace(),
                            tmp_out_array->ptr(),
                            phi::CPUPlace(),
                            reinterpret_cast<void *>(out_data.data()),
                            out_data.size() * sizeof(T *));

    T **out_grad_data = reinterpret_cast<T **>(tmp_out_array->ptr());
    ComputeKernelParameter(lod_length);
    PartialSumGradCUDAKernel<T>
        <<<grids, blocks, 0, stream>>>(out_grad_data,
                                       out_grad.data<T>(),
                                       lod_length,
                                       out_data.size(),
                                       start_index,
                                       length,
                                       row_length);
  }
}
}  // namespace phi
