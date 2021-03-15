// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_runtime.h>
#include <cassert>
#include <limits>
#include "paddle/fluid/operators/log_softmax_op.h"

namespace paddle {
namespace operators {

#define WARP_SIZE 32
#define LANE_MASK 0xffffffff

const int max_threads = 1024;

int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template <typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};
template <typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};
template <typename T, int WARP_BATCH, int KERNEL_WARP_SIZE,
          template <typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(T* sum) {
  ReduceOp<T> op;
#pragma unroll
  for (int mask = KERNEL_WARP_SIZE / 2; mask > 0; mask >>= 1) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      T b = __shfl_xor_sync(LANE_MASK, sum[i], mask, KERNEL_WARP_SIZE);
      sum[i] = op(sum[i], b);
    }
  }
}

template <typename T>
struct LogSoftMaxForwardEpilogue {
  __device__ __forceinline__ LogSoftMaxForwardEpilogue(T max_input, T sum)
      : max_input(max_input), logsum(std::log(sum)) {}

  __device__ __forceinline__ T operator()(T input) const {
    return static_cast<T>(input - max_input - logsum);
  }

  const T max_input;
  const T logsum;
};

inline dim3 SoftMax_getBlockSize(int ILP, uint64_t dim_size) {
  uint64_t block_size = 1;
  uint64_t max_block_size =
      std::min(dim_size / ILP, static_cast<uint64_t>(max_threads));

  // In the vectorized case we want to trade off allowing more of the buffers to
  // be accessed
  // in a vectorized way against wanting a larger block size to get better
  // utilisation.
  // In general with ILP you can have (ILP-1)/ILP of the buffer accessed
  // vectorised, at the risk
  // of having a very small block size. We choose to keep >= 1/2 of the buffer
  // vectorised while
  // allowing a larger block size.
  if (ILP > 1) {
    max_block_size /= 2;
  }

  while (block_size < (max_block_size)) block_size *= 2;
  // Launch at least a single warp - the kernel assumes that.
  block_size = std::max(block_size, static_cast<uint64_t>(WARP_SIZE));
  return dim3(block_size);
}

inline dim3 SpatialSoftMax_getBlockSize(uint64_t outer_size, uint64_t dim_size,
                                        uint64_t inner_size) {
  uint32_t inner_threads = inner_size;
  inner_threads = std::min(inner_threads, static_cast<uint32_t>(max_threads));
  uint32_t dim_threads = 1;
  if (inner_threads <= 64 && dim_size >= 64) {
    while (inner_threads * dim_threads <= max_threads &&
           dim_threads <= dim_size)
      dim_threads *= 2;
    dim_threads /= 2;
  }
  printf("[getBlockSize] dim_threads: %d inner_threads: %d\n", dim_threads,
         inner_threads);
  return dim3(dim_threads, inner_threads);
}

inline dim3 SpatialSoftMax_getGridSize(dim3 block, uint32_t max_active_blocks,
                                       uint64_t outer_size, uint64_t dim_size,
                                       uint64_t inner_size) {
  // First, tile as many blocks as we can over the y axis
  uint32_t inner_blocks = (inner_size + block.y - 1) / block.y;
  if (inner_blocks > max_active_blocks) inner_blocks = max_active_blocks;
  // Fill the x axis with as many blocks as we can fit (a little more is ok too)
  uint32_t outer_blocks = (max_active_blocks + inner_blocks - 1) / inner_blocks;
  if (outer_blocks > outer_size) outer_blocks = outer_size;
  printf("[getGridSize] outer_blocks: %d, inner_blocks: %d\n", outer_blocks,
         inner_blocks);
  return dim3(outer_blocks, inner_blocks);
}

template <typename T, template <typename> class ReduceOp>
__forceinline__ __device__ T spatialBlockReduceX(T* shared, T val) {
  ReduceOp<T> op;
  shared += threadIdx.y * blockDim.x;

  __syncthreads();

  shared[threadIdx.x] = val;

  // NOTE: loop starts with __syncthreads()
  int offset = blockDim.x / 2;
  while (offset > 0) {
    __syncthreads();
    if (threadIdx.x < offset)
      shared[threadIdx.x] =
          op(shared[threadIdx.x], shared[threadIdx.x + offset]);
    offset /= 2;
  }

  __syncthreads();

  return shared[0];
}

template <typename T, template <typename> class Epilogue>
__global__ void cunn_SpatialSoftMaxForward(T* output, const T* input,
                                           int outer_size, int dim_size,
                                           int inner_size) {
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<T*>(smem);
  const uint32_t outer_stride = inner_size * dim_size;
  const uint32_t dim_stride = inner_size;

  for (uint32_t outer_index = blockIdx.x; outer_index < outer_size;
       outer_index += gridDim.x) {
    const uint32_t outer_offset = outer_index * outer_stride;
    for (uint32_t inner_index = blockIdx.y * blockDim.y + threadIdx.y;
         inner_index < inner_size; inner_index += blockDim.y * gridDim.y) {
      const uint32_t data_offset = outer_offset + inner_index;
      ////////////////////////////////////////////////////////////
      // These two blocks are really equivalent, but specializing on
      // blockDim.x == 1 makes the kernel faster when it's unused.
      // I didn't want to thread an extra template parameter, and nvcc
      // seems to be smart enough to hoist the if outside of the loops.
      ////////////////////////////////////////////////////////////

      if (blockDim.x > 1) {
        T max_input = std::numeric_limits<T>::lowest();
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
          const T value = static_cast<T>(input[data_offset + d * dim_stride]);
          max_input = Max<T>()(max_input, value);
        }
        max_input = spatialBlockReduceX<T, Max>(sdata, max_input);

        T sum = 0;
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          sum += std::exp(static_cast<T>(input[data_offset + d * dim_stride]) -
                          max_input);
        sum = spatialBlockReduceX<T, Add>(sdata, sum);

        Epilogue<T> epilogue(max_input, sum);
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          output[data_offset + d * dim_stride] =
              epilogue(input[data_offset + d * dim_stride]);
      } else {
        T max_input = std::numeric_limits<T>::lowest();
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
          const T value = static_cast<T>(input[data_offset + d * dim_stride]);
          max_input = Max<T>()(max_input, value);
        }
        T sum = 0;
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          sum += std::exp(static_cast<T>(input[data_offset + d * dim_stride]) -
                          max_input);
        Epilogue<T> epilogue(max_input, sum);
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          output[data_offset + d * dim_stride] =
              epilogue(input[data_offset + d * dim_stride]);
      }
    }
  }
}

template <typename T, typename Kernel>
void SpatialSoftMax_getLaunchSizes(Kernel k, uint64_t outer_size,
                                   uint64_t dim_size, uint64_t inner_size,
                                   dim3& grid, dim3& block,
                                   uint32_t& smem_size) {
  block = SpatialSoftMax_getBlockSize(outer_size, dim_size, inner_size);
  uint32_t block_threads = block.x * block.y;
  smem_size = block.x == 1 ? 0 : block_threads * sizeof(T);
  int max_active_blocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, k,
                                                block_threads, smem_size);
  cudaDeviceProp prop;
  max_active_blocks *= prop.multiProcessorCount;
  grid = SpatialSoftMax_getGridSize(block, max_active_blocks, outer_size,
                                    dim_size, inner_size);
  printf(
      "[getLaunchSizes] outer_size: %d dim_size: %d inner_size: %d smem_size: "
      "%d\n",
      outer_size, dim_size, inner_size, smem_size);
}

template <typename T, int log2_elements>
__global__ void softmax_warp_forward(T* dst, const T* src,
                                     int batch_size, /*outer_size*/
                                     int stride,     /*dim_size*/
                                     int element_count /*dim_size*/) {
  // KERNEL_WARP_SIZE and WARP_BATCH must match the return values
  // batches_per_warp and warp_size of method warp_softmax_forward_kernel.
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int KERNEL_WARP_SIZE =
      (next_power_of_two < WARP_SIZE) ? next_power_of_two : WARP_SIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / KERNEL_WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
  //  printf("first_batch= %d blockDim.y * blockIdx.x + threadIdx.y: %d * %d +
  //  %d \n", first_batch, blockDim.y, blockIdx.x, threadIdx.y);

  // batch_size might not be a multiple of WARP_BATCH. Check how
  // many batches have to computed within this WARP.
  int local_batches = batch_size - first_batch;  // batch_size=3
  if (local_batches > WARP_BATCH) local_batches = WARP_BATCH;
  //  printf("local_batches: %d\n", local_batches);

  // there might be multiple batches per warp. compute the index within the
  // batch
  int local_idx = threadIdx.x;

  src += first_batch * stride + local_idx;
  dst += first_batch * stride + local_idx;

  // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified
  // to one loop,
  // but I think doing so would obfuscate the logic of the algorithm, thus I
  // chose to keep
  // the nested loops.
  // This should have no impact on performance because the loops are unrolled
  // anyway.

  // 1.load data from global memory
  T elements[WARP_BATCH][WARP_ITERATIONS];  // [2][1], i:0~1, it:0 .
  //  printf("WARP_BATCH: %d - WARP_ITERATIONS: %d\n", WARP_BATCH,
  //  WARP_ITERATIONS);
  int idx = threadIdx.x + blockDim.x * threadIdx.y;

  //  printf("idx: %d", idx);
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * KERNEL_WARP_SIZE;
      if (element_index < batch_element_count) {
        elements[i][it] = src[i * element_count + it * KERNEL_WARP_SIZE];
        // printf("idx: %d elements[i][it]: [%d][%d] %f \n", idx, i, it,
        // elements[i][it]);
      } else {
        elements[i][it] = -std::numeric_limits<T>::infinity();
        // printf("idx: %d elements[i][it]: [%d][%d] %f \n", idx, i, it,
        // elements[i][it]);
      }
    }
  }
  printf("idx %d - elem1: %f elem2: %f elem3: %f elem4: %f\n", idx,
         elements[0][0], elements[1][0], elements[0][1], elements[1][1]);

  // 2.compute max_value
  T max_value[WARP_BATCH];  // WARP_BATCH=2
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  //  printf("idx: %d - elem1: %f elem2: %f max_value1: %f max_value2: %f \n",
  //  idx, elements[0][0], elements[1][0], max_value[0], max_value[1]);
  warp_reduce<T, WARP_BATCH, KERNEL_WARP_SIZE, Max>(max_value);
  //  printf("idx: %d - elem1: %f elem2: %f max_value1: %f max_value2: %f \n",
  //  idx, elements[0][0], elements[1][0], max_value[0], max_value[1]);

  T sum[WARP_BATCH]{0.0f};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      sum[i] += std::exp(elements[i][it] - max_value[i]);
    }
  }
  // printf("idx: %d - elem1: %f elem2: %f sum1: %f sum2: %f \n", idx,
  // elements[0][0], elements[1][0], sum[0], sum[1]);
  warp_reduce<T, WARP_BATCH, KERNEL_WARP_SIZE, Add>(sum);
// printf("idx: %d - elem1: %f elem2: %f sum1: %f sum2: %f \n", idx,
// elements[0][0], elements[1][0], sum[0], sum[1]);

// 3.store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches) break;
    sum[i] = std::log(sum[i]);
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * KERNEL_WARP_SIZE;
      if (element_index < element_count) {
        dst[i * element_count + it * KERNEL_WARP_SIZE] =
            elements[i][it] - max_value[i] - sum[i];
      } else {
        break;
      }
    }
  }
  // printf("idx: %d - elem1: %f elem2: %f res1: %f res2: %f\n", idx,
  // elements[0][0], elements[1][0], elements[0][0]-max_value[0]-sum[0],
  // elements[1][0]-max_value[1]-sum[1]);
}

template <typename T>
void dispatch_softmax_forward(T* dst, const T* src,
                              int softmax_elements,        /*dim_size*/
                              int softmax_elements_stride, /*dim_size*/
                              int batch_count /*outer_size*/) {
  assert(softmax_elements >= 0 && softmax_elements <= 1024);
  if (softmax_elements == 0) {
    return;
  } else {
    printf("***** dispatch_softmax_forward BEGIN *****\n");
    int log2_elements = log2_ceil(softmax_elements);
    printf("*dim_size: %d\n*log2_elements: %d\n", softmax_elements,
           log2_elements);
    const int next_power_of_two = 1 << log2_elements;
    printf("*next_power_of_two: %d\n\n", next_power_of_two);

    // This value must match the WARP_SIZE constexpr value computed inside
    // softmax_warp_forward.
    int warp_size =
        (next_power_of_two < WARP_SIZE) ? next_power_of_two : WARP_SIZE;
    printf("warp_size(KERNEL_WARP_SIZE): %d\n", warp_size);

    // This value must match the WARP_BATCH constexpr value computed inside
    // softmax_warp_forward.
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
    printf("batches_per_warp(WARP_BATCH): %d\n", batches_per_warp);

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    printf("warps_per_block: %d\n", warps_per_block);
    printf("batches_per_block: %d\n", batches_per_block);
    printf("blocks: %d\n", blocks);
    dim3 threads(warp_size, warps_per_block, 1);
    printf("***** dispatch_softmax_forward END *****\n");

    printf("***** softmax_warp_forward BEGIN *****\n");

    switch (log2_elements) {
#define LAUNCH_SOFTMAX_WARP_FORWARD(L2E)                                   \
  case L2E:                                                                \
    softmax_warp_forward<T, L2E><<<blocks, threads, 0>>>(                  \
        dst, src, batch_count, softmax_elements_stride, softmax_elements); \
    break;

      LAUNCH_SOFTMAX_WARP_FORWARD(0);  // 1
      LAUNCH_SOFTMAX_WARP_FORWARD(1);  // 2
      LAUNCH_SOFTMAX_WARP_FORWARD(2);  // 4
      LAUNCH_SOFTMAX_WARP_FORWARD(3);  // 8
      LAUNCH_SOFTMAX_WARP_FORWARD(4);  // 16
      LAUNCH_SOFTMAX_WARP_FORWARD(5);  // 32
      LAUNCH_SOFTMAX_WARP_FORWARD(6);  // 64
      LAUNCH_SOFTMAX_WARP_FORWARD(7);  // 128
      LAUNCH_SOFTMAX_WARP_FORWARD(8);  // 256
      LAUNCH_SOFTMAX_WARP_FORWARD(9);  // 512
      LAUNCH_SOFTMAX_WARP_FORWARD(10);
      ;  // 1024
      default:
        break;
    }
    printf("***** softmax_warp_forward END *****\n");
  }
}

template <typename DeviceContext, typename T,
          template <typename> class Epilogue>
struct LogSoftmaxCUDAFunctor {
  void operator()(const DeviceContext& context, const framework::Tensor* X,
                  framework::Tensor* Out, const int axis) {
    // printf("------------ log-softmax-op cuda version BEGIN\n");
    // printf("--------- axis: %d\n", axis);
    int along_axis = (axis < 0) ? axis + X->dims().size() : axis;
    // printf("--------- along_axis: %d\n", along_axis);
    int outer_size = 1;
    int dim_size = X->dims()[along_axis];
    printf("--------- dim_size: %d\n", dim_size);
    printf("--------- X->numel(): %d\n", X->numel());

    const auto* input_data = X->data<T>();
    auto* output_data = Out->mutable_data<T>(context.GetPlace());

    if (X->numel() > 0) {
      int inner_size = 1;
      for (int i = 0; i < along_axis; i++) outer_size *= X->dims()[i];
      for (int i = along_axis + 1; i < X->dims().size(); i++)
        inner_size *= X->dims()[i];
      cudaStream_t stream = context.stream();
      printf("---- outer_size: %d, inner_size: %d, \n", outer_size, inner_size);

      if (inner_size == 1) {
        dim3 grid(outer_size);

        if (dim_size <= 1024 && dim_size * sizeof(T) <= 4096) {
          dispatch_softmax_forward<T>(output_data, input_data, dim_size,
                                      dim_size, outer_size);
        } else {
          printf("when !(dim_size <= 1024 && dim_size*sizeof(T) <= 4096) \n");
          // constexpr int ilp = sizeof(float4)/sizeof(T);
          // dim3 block = SoftMax_getBlockSize(ilp, dim_size);
          // cunn_SoftMaxForward<ilp,T,T,T,Epilogue><<<grid, block,
          // block.x*sizeof(T), stream>>>(output_data, input_data, dim_size);
        }
      } else {  // 当 axis!=-1 时
        uint32_t smem_size;
        dim3 grid, block;
        // calculate grid, block & smem
        SpatialSoftMax_getLaunchSizes<T>(
            &cunn_SpatialSoftMaxForward<T, Epilogue>, outer_size, dim_size,
            inner_size, grid, block, smem_size);
        printf("Launch kernel\n");
        cunn_SpatialSoftMaxForward<
            T, Epilogue><<<grid, block, smem_size, stream>>>(
            output_data, input_data, outer_size, dim_size, inner_size);
      }
    }
    // printf("------------ log-softmax-op cuda version END\n");
  }
};

template <typename T>
class LogSoftmaxKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* X = context.Input<framework::Tensor>("X");
    auto* Out = context.Output<framework::Tensor>("Out");
    const int rank = X->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);

    // printf("---- X->dims().size(): %d\n", X->dims().size());
    // printf("---- X->numel(): %d\n", X->numel());
    // printf("---- X->dims()[0]: %d, X->dims()[1]: %d, X->dims()[2]: %d\n",
    // X->dims()[0], X->dims()[1], X->dims()[2]);

    // const auto* input_data = X->data<T>();
    // auto* output_data = Out->mutable_data<T>(context.GetPlace());
    Out->mutable_data<T>(context.GetPlace());
    LogSoftmaxCUDAFunctor<platform::CUDADeviceContext, T,
                          LogSoftMaxForwardEpilogue>()(
        context.template device_context<platform::CUDADeviceContext>(), X, Out,
        axis);
  }
};

}  // operators
}  // paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    log_softmax, ops::LogSoftmaxKernel<plat::CUDADeviceContext, float>,
    ops::LogSoftmaxKernel<plat::CUDADeviceContext, double> /*,
    ops::LogSoftmaxKernel<plat::CUDADeviceContext, plat::float16>*/);
REGISTER_OP_CUDA_KERNEL(
    log_softmax_grad, ops::LogSoftmaxGradKernel<plat::CUDADeviceContext, float>,
    ops::LogSoftmaxGradKernel<plat::CUDADeviceContext, double> /*,
    ops::LogSoftmaxGradKernel<plat::CUDADeviceContext, plat::float16>*/);
