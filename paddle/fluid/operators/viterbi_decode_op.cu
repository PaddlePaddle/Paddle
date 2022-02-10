/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/elementwise/elementwise_functor.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/viterbi_decode_op.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

namespace paddle {
namespace operators {

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

int64_t ComputeBlockSize(int64_t col) {
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
}

template <template <typename T> typename BinaryFunctor, typename T>
struct BinaryOperation<platform::CUDADeviceContext, BinaryFunctor, T> {
  void operator()(const platform::CUDADeviceContext& dev_ctx, const Tensor& lhs,
                  const Tensor& rhs, Tensor* output) {
    std::vector<const Tensor*> ins{&lhs, &rhs};
    std::vector<Tensor*> outs{output};
    paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T,
                                                   T>(dev_ctx, ins, &outs, -1,
                                                      BinaryFunctor<T>());
  }
};

template <template <typename InT, typename OutT> typename CompareFunctor,
          typename T>
struct GetMask<platform::CUDADeviceContext, CompareFunctor, T> {
  void operator()(const framework::ExecutionContext& ctx, const Tensor& lhs,
                  const Tensor& rhs, Tensor* mask) {
    std::vector<const Tensor*> ins = {&lhs, &rhs};
    std::vector<Tensor*> outs = {mask};
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    paddle::operators::LaunchSameDimsElementwiseCudaKernel<T>(
        dev_ctx, ins, &outs, CompareFunctor<int64_t, T>());
  }
};

template <typename T, typename IndType, size_t BlockDim>
__global__ void ArgmaxCUDAKernel(const int64_t height,     // n * h
                                 const int64_t width,      // c
                                 const int64_t post_size,  // h
                                 const T* in, IndType* out_idx, T* out) {
  typedef cub::BlockReduce<cub::KeyValuePair<int, T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  cub::ArgMax reducer;
  T init = (std::numeric_limits<T>::lowest)();  // for windows compile
  for (int idx = blockIdx.x; idx < height; idx += gridDim.x) {
    cub::KeyValuePair<int, T> kv_pair = {-1, init};
    int h = idx / post_size;
    int w = idx % post_size;
    for (int k = threadIdx.x; k < width; k += blockDim.x) {
      kv_pair =
          reducer({k, in[h * width * post_size + k * post_size + w]}, kv_pair);
    }
    kv_pair = BlockReduce(temp_storage).Reduce(kv_pair, reducer);
    if (threadIdx.x == 0) {
      // return max, argmax
      if (out_idx != nullptr) out_idx[idx] = static_cast<IndType>(kv_pair.key);
      if (out != nullptr) out[idx] = kv_pair.value;
    }
    __syncthreads();
  }
}

__global__ void ARangeKernel(int64_t* data, int num, int64_t scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int start = idx; idx < num; idx += gridDim.x) {
    data[idx] = idx * scale;
  }
}

template <>
struct ARange<platform::CUDADeviceContext> {
  void operator()(const platform::CUDADeviceContext& dev_ctx, int64_t* data,
                  int num, int64_t scale) {
    int64_t kBlockDim = ComputeBlockSize(num);
    // kBlockDim > num at most of time, so we can set grid = 1
    ARangeKernel<<<1, kBlockDim, 0, dev_ctx.stream()>>>(data, num, scale);
  }
};

template <typename T, typename IndType>
struct Argmax<platform::CUDADeviceContext, T, IndType> {
  void operator()(const framework::ExecutionContext& ctx, const Tensor& input,
                  Tensor* out_idx, Tensor* out, int axis) {
    framework::DDim input_dims = input.dims();
    int64_t numel = input.numel();
    int64_t groups = numel / input_dims[axis];
    int64_t pre = 1;
    int64_t post = 1;
    int64_t n = input_dims[axis];
    for (int i = 0; i < axis; i++) {
      pre *= input_dims[i];
    }
    for (int i = axis + 1; i < input_dims.size(); i++) {
      post *= input_dims[i];
    }
    const auto& dev_ctx = ctx.cuda_device_context();
    auto cu_stream = dev_ctx.stream();
    int64_t max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize()[0];
    int64_t height = pre * post;
    int64_t width = n;
    int64_t grid_size = height < max_grid_dimx ? height : max_grid_dimx;
    const T* in_data = input.data<T>();
    IndType* out_idx_data = out_idx->data<IndType>();
    T* out_data = out->data<T>();
    switch (ComputeBlockSize(width)) {
      FIXED_BLOCK_DIM_CASE(
          ArgmaxCUDAKernel<T, IndType,
                           kBlockDim><<<grid_size, kBlockDim, 0, cu_stream>>>(
              height, width, post, in_data, out_idx_data, out_data));
    }
  }
};

template <typename T>
struct GetMaxValue<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& dev_ctx,
                  const Tensor& input, T* max_value) {
    Tensor out_data;
    out_data.Resize(framework::make_ddim({1}));
    out_data.mutable_data<T>(platform::CUDAPlace());
    switch (ComputeBlockSize(input.numel())) {
      FIXED_BLOCK_DIM_CASE(
          ArgmaxCUDAKernel<T, T,
                           kBlockDim><<<1, kBlockDim, 0, dev_ctx.stream()>>>(
              1, input.numel(), 1, input.data<int64_t>(), nullptr,
              out_data.data<int64_t>()));
    }
    Tensor max_value_tensor;
    framework::TensorCopy(out_data, platform::CPUPlace(), &max_value_tensor);
    *max_value = max_value_tensor.data<T>()[0];
  }
};

template <typename T, typename IndexT>
struct Gather<platform::CUDADeviceContext, T, IndexT> {
  void operator()(const platform::CUDADeviceContext& ctx, const Tensor& src,
                  const Tensor& index, Tensor* output) {
    GPUGather<T, IndexT>(ctx, src, index, output);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace platform = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    viterbi_decode,
    ops::ViterbiDecodeKernel<platform::CUDADeviceContext, float>,
    ops::ViterbiDecodeKernel<platform::CUDADeviceContext, double>);
