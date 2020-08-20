/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#ifdef __NVCC__

#include <cub/cub.cuh>
#include <limits>
#include <string>
#include <typeinfo>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

namespace {  // NOLINT
template <typename K, typename V>
using KeyValuePair = cub::KeyValuePair<K, V>;
using Tensor = framework::Tensor;

}  // end namespace

#define FIXED_BLOCK_DIM_CASE_BASE(log2_block_dim, ...)  \
  case (1 << (log2_block_dim)): {                       \
    constexpr auto kBlockDim = (1 << (log2_block_dim)); \
    __VA_ARGS__;                                        \
  } break

#define FIXED_BLOCK_DIM_CASE(...)               \
  FIXED_BLOCK_DIM_CASE_BASE(10, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(9, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(8, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(7, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(6, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(5, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(4, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(3, ##__VA_ARGS__);

template <typename T, typename IndType, class Reducer, size_t BlockDim>
__global__ void ArgCUDAKernel(const IndType height,     // n * h
                              const IndType width,      // c
                              const IndType post_size,  // h
                              const Reducer reducer, const T init, const T* in,
                              IndType* out) {
  typedef cub::BlockReduce<KeyValuePair<int, T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int idx = blockIdx.x; idx < height; idx += gridDim.x) {
    KeyValuePair<int, T> kv_pair = {-1, init};
    int h = idx / post_size;
    int w = idx % post_size;
    for (int k = threadIdx.x; k < width; k += blockDim.x) {
      kv_pair =
          reducer({k, in[h * width * post_size + k * post_size + w]}, kv_pair);
    }
    kv_pair = BlockReduce(temp_storage).Reduce(kv_pair, reducer);
    if (threadIdx.x == 0) {
      out[idx] = static_cast<IndType>(kv_pair.key);
    }
    __syncthreads();
  }
}

template <typename T, typename IndType, class Reducer>
void ComputeFullArg(const platform::CUDADeviceContext& ctx, const Tensor& input,
                    Tensor* indices, const IndType pre, const IndType post,
                    const IndType n) {
  auto cu_stream = ctx.stream();
  auto ComputeBlockSize = [](IndType col) {
    if (col > 512)
      return 1024;
    else if (col > 256)
      return 512;
    else if (col > 128)
      return 256;
    else if (col > 64)
      return 128;
    else if (col > 32)
      return 64;
    else if (col > 16)
      return 32;
    else if (col > 8)
      return 16;
    else
      return 8;
  };

  int max_grid_dimx = ctx.GetCUDAMaxGridDimSize().x;
  int height = pre * post;
  int width = n;
  int grid_size = height < max_grid_dimx ? height : max_grid_dimx;

  const T* in_data = input.data<T>();
  IndType* out_data = indices->mutable_data<IndType>(ctx.GetPlace());

  if (typeid(Reducer) == typeid(cub::ArgMax)) {
    switch (ComputeBlockSize(width)) {
      FIXED_BLOCK_DIM_CASE(
          ArgCUDAKernel<T, IndType, Reducer,
                        kBlockDim><<<grid_size, kBlockDim, 0, cu_stream>>>(
              height, width, post, Reducer(), std::numeric_limits<T>::lowest(),
              in_data, out_data));
    }
  } else {
    switch (ComputeBlockSize(width)) {
      FIXED_BLOCK_DIM_CASE(
          ArgCUDAKernel<T, IndType, Reducer,
                        kBlockDim><<<grid_size, kBlockDim, 0, cu_stream>>>(
              height, width, post, Reducer(), std::numeric_limits<T>::max(),
              in_data, out_data));
    }
  }
}

template <typename T, class Reducer>
class ArgMinMaxOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    int axis = ctx.Attr<int64_t>("axis");
    auto in_dims = input->dims();
    axis = (axis < 0) ? (in_dims.size() + axis) : axis;

    int64_t numel = input->numel();
    int64_t groups = numel / in_dims[axis];
    int64_t pre = 1;
    int64_t post = 1;
    int64_t n = in_dims[axis];

    for (int i = 0; i < axis; i++) {
      pre *= in_dims[i];
    }

    for (int i = axis + 1; i < in_dims.size(); i++) {
      post *= in_dims[i];
    }

    const auto& dev_ctx = ctx.cuda_device_context();
    ComputeFullArg<T, int64_t, Reducer>(dev_ctx, *input, output, pre, post, n);
  }
};

#endif

}  // namespace operators
}  // namespace paddle
