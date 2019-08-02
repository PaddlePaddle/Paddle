/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <unordered_set>
#include <vector>
#include "math/math_function.h"
#include "paddle/fluid/framework/dim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
template <typename T, typename IndexT = int>
__global__ void ScatterNDCUDAKernel(const IndexT totalE, const T* param,
                                    IndexT* param_stride, const IndexT* indices,
                                    IndexT* indices_dim, IndexT* indices_stride,
                                    T* output, int axis, const IndexT ddims) {
  CUDA_1D_KERNEL_LOOP(i, totalE) {
    IndexT indices_offset = 0;
    IndexT src_offset = 0;
    IndexT output_offset = 0;

    for (int d = ddims - 1; d >= 0; --d) {
      int indices_i = i % indices_dim[d];
      indices_offset += indices_i * indices_stride[d];
      src_offset += indices_i * indices_stride[d];
      if (d != axis) output_offset += indices_i * param_stride[d];
      i = i / indices_dim[d];
    }
    IndexT indices_value = indices[indices_offset];
    assert(indices_value >= 0 && indices_value < param->dims()[axis]);
    output_offset += indices_value * param_stride[axis];
    printf("src_offset: %d\n", src_offset);
    printf("output_offset: %d\n", output_offset);
    if (output_offset == 102) printf("sdadsada: %d\n", param[src_offset]);
    output[output_offset] = param[src_offset];
  }
}

template <typename DeviceContext, typename T, typename IndexT = int>
void GPUScatterNDAssign(const framework::ExecutionContext& context,
                        const Tensor& input, const Tensor& src,
                        const Tensor& index, Tensor* output, int axis) {
  const auto& ctx = context.template device_context<DeviceContext>();
  const auto gplace = boost::get<platform::CUDAPlace>(ctx.GetPlace());
  auto cplace = platform::CPUPlace();
  const T* p_src = src.data<T>();
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();

  const auto index_dims = index.dims();
  int ddims = index_dims.size();
  const auto src_stride = framework::stride(input.dims());
  const auto index_stride = framework::stride(index_dims);

  std::vector<int> index_ddims(ddims);
  std::vector<int> src_sstride(ddims);
  std::vector<int> index_sstride(ddims);

  for (int i = 0; i < ddims; ++i) {
    index_ddims[i] = static_cast<int>(index_dims[i]);
    src_sstride[i] = static_cast<int>(src_stride[i]);
    index_sstride[i] = static_cast<int>(index_stride[i]);
  }

  // set block and grid num
  const IndexT block = 512;
  const IndexT n = framework::product(index_dims);
  const IndexT grid = (n + block - 1) / block;

  auto& dev_ctx = context.cuda_device_context();
  auto& allocator = platform::DeviceTemporaryAllocator::Instance().Get(dev_ctx);
  int bytes = ddims * sizeof(int);

  auto dims_ptr = allocator.Allocate(bytes);
  int* g_index_dim = reinterpret_cast<int*>(dims_ptr->ptr());
  memory::Copy(gplace, g_index_dim, cplace, index_ddims.data(), bytes,
               ctx.stream());

  auto src_stride_ptr = allocator.Allocate(bytes);
  int* g_src_stride = reinterpret_cast<int*>(src_stride_ptr->ptr());
  memory::Copy(gplace, g_src_stride, cplace, src_sstride.data(), bytes,
               ctx.stream());

  auto index_stride_ptr = allocator.Allocate(bytes);
  int* g_index_stride = reinterpret_cast<int*>(index_stride_ptr->ptr());
  memory::Copy(gplace, g_index_stride, cplace, index_sstride.data(), bytes,
               ctx.stream());

  ScatterNDCUDAKernel<T, IndexT><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
      n, p_src, g_src_stride, p_index, g_index_dim, g_index_stride, p_output,
      axis, ddims);
}

}  // namespace operators
}  // namespace paddle
