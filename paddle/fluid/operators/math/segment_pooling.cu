/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/segment_pooling.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_launch_config.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, typename Index, int DimTileSize>
__global__ void SegmentMeanCustomKernel(
    const Index* segment_ids, const T* input, T* output, T* summed_ids,
    const Index input_length_size, const Index inner_dim_size,
    const Index output_length_size, const Index total_stripe_count) {
  CUDA_KERNEL_LOOP(stripe_index, total_stripe_count) {
    const Index segment_offset = stripe_index % inner_dim_size;
    const Index dim_index_base =
        stripe_index / inner_dim_size * Index(DimTileSize);
    const Index actual_height =
        min(Index(DimTileSize), input_length_size - dim_index_base);

    Index first_segment_id = segment_ids[dim_index_base];
    Index last_segment_id = -1;
    if (dim_index_base > 0) {
      last_segment_id = segment_ids[dim_index_base - 1];
    }
    if (segment_offset == 0) {
      T sum = T(0);
      for (Index j = 0; j < actual_height; j++) {
        Index current_segment_id = segment_ids[dim_index_base + j];
        // Note(ZHUI): following check may cause
        // cudaErrorLaunchOutOfResources.
        // PADDLE_ENFORCE(current_segment_id >= last_segment_id,
        //               "the segment ids should be sorted, but got "
        //               "segment_ids[%d]:%d > segment_ids[%d]:%d.",
        //               dim_index_base + j - 1, dim_index_base + j,
        //               last_segment_id, current_segment_id);

        if (j > 0 && current_segment_id > last_segment_id) {
          if (last_segment_id == first_segment_id) {
            platform::CudaAtomicAdd(summed_ids + last_segment_id, sum);
          } else {
            *(summed_ids + last_segment_id) = sum;
          }
          sum = T(0);
        }
        sum += T(1);
        last_segment_id = current_segment_id;
      }
      platform::CudaAtomicAdd(summed_ids + last_segment_id, sum);
    }
    // ensure last_segment_id is the largest
    last_segment_id = output_length_size;
    __syncthreads();
    T sum = T(0);
    for (Index j = 0; j < actual_height; j++) {
      Index current_segment_id = segment_ids[dim_index_base + j];
      if (current_segment_id > last_segment_id) {
        const Index output_index =
            last_segment_id * inner_dim_size + segment_offset;
        if (last_segment_id == first_segment_id) {
          platform::CudaAtomicAdd(output + output_index,
                                  sum / *(summed_ids + last_segment_id));
        } else {
          *(output + output_index) = sum / *(summed_ids + last_segment_id);
        }
        sum = T(0);
      }
      sum += input[(dim_index_base + j) * inner_dim_size + segment_offset];
      last_segment_id = current_segment_id;
    }
    const Index output_index =
        last_segment_id * inner_dim_size + segment_offset;
    platform::CudaAtomicAdd(output + output_index,
                            sum / *(summed_ids + last_segment_id));
  }
}

template <typename T, typename Index, typename Helper, typename Pool>
__global__ void SegmentOpsKernel(const Index* segment_ids, const T* input,
                                 T* output, Helper h, Pool pool) {
  CUDA_KERNEL_LOOP(stripe_index, h.total_stripe_count) {
    Index segment_offset, dim_index_base, actual_height;
    Index inner_dim_size = h.inner_dim_size;
    h.calculate(stripe_index, &segment_offset, &dim_index_base, &actual_height);

    T minmax = pool.initial();
    Index first_segment_id = segment_ids[dim_index_base];
    // -1 is for the start value when interval_id = 0
    Index last_segment_id = -1;
    if (dim_index_base > 0) {
      last_segment_id = segment_ids[dim_index_base - 1];
    }

    for (Index j = 0; j < actual_height; j++) {
      Index current_segment_id = segment_ids[dim_index_base + j];
      // ensure the segment_ids is sorted.
      PADDLE_ENFORCE(current_segment_id >= last_segment_id,
                     "The segment ids should be sorted, but got "
                     "segment_ids[%d]:%d > segment_ids[%d]:%d.",
                     dim_index_base + j - 1, dim_index_base + j,
                     last_segment_id, current_segment_id);

      if (current_segment_id > last_segment_id) {
        // reset the interval value which do not have corresponding ids.
        for (Index interval_id = last_segment_id + 1;
             interval_id < current_segment_id; ++interval_id) {
          *(output + interval_id * inner_dim_size + segment_offset) = 0;
        }
        // don't update result when j=0
        if (j > 0) {
          const Index output_index =
              last_segment_id * inner_dim_size + segment_offset;
          if (last_segment_id == first_segment_id) {
            pool.atomic(output + output_index, minmax);
          } else {
            *(output + output_index) = minmax;
          }
          minmax = pool.initial();
        }
      }
      pool.compute(
          input[(dim_index_base + j) * inner_dim_size + segment_offset],
          &minmax);
      last_segment_id = current_segment_id;
    }
    const Index output_index =
        last_segment_id * inner_dim_size + segment_offset;
    pool.atomic(output + output_index, minmax);
  }
}

template <typename T, typename Index, typename Helper>
__global__ void SegmentIndexGradKernel(const Index* segment_ids, const T* input,
                                       const T* output, const T* out_grad,
                                       T* in_grad, Helper h) {
  CUDA_KERNEL_LOOP(stripe_index, h.total_stripe_count) {
    Index segment_offset, dim_index_base, actual_height;
    h.calculate(stripe_index, &segment_offset, &dim_index_base, &actual_height);

    for (Index j = 0; j < actual_height; j++) {
      Index current_segment_id = segment_ids[dim_index_base + j];
      Index input_index =
          (dim_index_base + j) * h.inner_dim_size + segment_offset;
      Index output_index =
          current_segment_id * h.inner_dim_size + segment_offset;
      if (input[input_index] == output[output_index]) {
        in_grad[input_index] = out_grad[output_index];
      }
    }
  }
}

template <class T>
class MaxPool {
 public:
  DEVICE inline T initial() { return static_cast<T>(-FLT_MAX); }
  DEVICE inline void compute(const T& x, T* y) { *y = *y > x ? *y : x; }
  DEVICE inline T atomic(T* address, const T val) {
    return platform::CudaAtomicMax(address, val);
  }
};

template <class T>
class MinPool {
 public:
  DEVICE inline T initial() { return static_cast<T>(FLT_MAX); }
  DEVICE inline void compute(const T& x, T* y) { *y = *y < x ? *y : x; }
  DEVICE inline T atomic(T* address, const T val) {
    return platform::CudaAtomicMin(address, val);
  }
};

template <class T>
class SumPool {
 public:
  DEVICE inline T initial() { return static_cast<T>(0); }
  DEVICE inline void compute(const T& x, T* y) { *y = *y + x; }
  DEVICE inline T atomic(T* address, const T val) {
    return platform::CudaAtomicAdd(address, val);
  }
};

template <class T>
class ArrangeHelper {
 public:
  const T input_total_size;
  const T input_length_size;
  const T output_length_size;
  T inner_dim_size;
  T total_stripe_count;
  const T DimTileSize = 8;

  ArrangeHelper(T a, T b, T c)
      : input_total_size(a), input_length_size(b), output_length_size(c) {
    T input_outer_dim_num_stripe =
        (input_length_size + DimTileSize - 1) / DimTileSize;
    inner_dim_size = input_total_size / input_length_size;
    total_stripe_count = inner_dim_size * input_outer_dim_num_stripe;
  }

  DEVICE inline void calculate(T stripe_index, T* segment_offset,
                               T* dim_index_base, T* actual_height) {
    *segment_offset = stripe_index % inner_dim_size;
    *dim_index_base = stripe_index / inner_dim_size * DimTileSize;
    *actual_height = min(DimTileSize, input_length_size - *dim_index_base);
  }
};

template <typename T, typename Index>
void SegmentPoolCUDAGradFunctor(const platform::CUDADeviceContext& ctx,
                                const framework::Tensor& input,
                                const framework::Tensor& segment_ids,
                                const framework::Tensor& output,
                                const framework::Tensor& out_grad,
                                framework::Tensor* in_grad,
                                const std::string pooltype = "SUM") {
  auto h = ArrangeHelper<Index>(input.numel(), segment_ids.dims()[0],
                                output.dims()[0]);
  auto config = platform::GetGpuLaunchConfig1D(ctx, h.total_stripe_count);
  if (pooltype == "MAX" || pooltype == "MIN") {
    SegmentIndexGradKernel<T, Index, ArrangeHelper<Index>><<<
        config.block_per_grid.x, config.thread_per_block.x, 0, ctx.stream()>>>(
        segment_ids.data<Index>(), input.data<T>(), output.data<T>(),
        out_grad.data<T>(), in_grad->data<T>(), h);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Unsupported segment pooling grad operation, Only MAX, MIN "
        "available, but got %s.",
        pooltype));
  }
}

template <typename T>
__global__ void SimpleDiv(T* x, const T* y, const int len, const int dim) {
  for (int i = blockIdx.x; i < len; i += gridDim.x) {
    __shared__ T y_i;
    auto base = i * dim;
    if (threadIdx.x == 0) {
      y_i = y[i];
    }
    __syncthreads();
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
      x[base + j] /= y_i;
    }
  }
}

template <typename T, typename IndexT>
class SegmentPoolFunctor<platform::CUDADeviceContext, T, IndexT> {
 public:
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& input,
                  const framework::Tensor& segment_ids,
                  framework::Tensor* output,
                  framework::Tensor* summed_ids = nullptr,
                  const std::string pooltype = "SUM") {
    auto h = ArrangeHelper<IndexT>(input.numel(), segment_ids.dims()[0],
                                   output->dims()[0]);
    auto config = platform::GetGpuLaunchConfig1D(ctx, h.total_stripe_count);
    if (pooltype == "MEAN") {
      SegmentMeanCustomKernel<
          T, IndexT, IndexT(8)><<<config.block_per_grid.x,
                                  config.thread_per_block.x, 0, ctx.stream()>>>(
          segment_ids.data<IndexT>(), input.data<T>(), output->data<T>(),
          summed_ids->data<T>(), h.input_length_size, h.inner_dim_size,
          h.output_length_size, h.total_stripe_count);
    } else if (pooltype == "SUM") {
      SumPool<T> pool;
      SegmentOpsKernel<
          T, IndexT, ArrangeHelper<IndexT>,
          SumPool<T>><<<config.block_per_grid.x, config.thread_per_block.x, 0,
                        ctx.stream()>>>(segment_ids.data<IndexT>(),
                                        input.data<T>(), output->data<T>(), h,
                                        pool);
    } else if (pooltype == "MAX") {
      MaxPool<T> pool;
      SegmentOpsKernel<
          T, IndexT, ArrangeHelper<IndexT>,
          MaxPool<T>><<<config.block_per_grid.x, config.thread_per_block.x, 0,
                        ctx.stream()>>>(segment_ids.data<IndexT>(),
                                        input.data<T>(), output->data<T>(), h,
                                        pool);
    } else if (pooltype == "MIN") {
      MinPool<T> pool;
      SegmentOpsKernel<
          T, IndexT, ArrangeHelper<IndexT>,
          MinPool<T>><<<config.block_per_grid.x, config.thread_per_block.x, 0,
                        ctx.stream()>>>(segment_ids.data<IndexT>(),
                                        input.data<T>(), output->data<T>(), h,
                                        pool);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported segment pooling operation, Only MEAN, SUM, MAX, MIN "
          "available, but got %s.",
          pooltype));
    }
  }
};

template <typename T, typename IndexT>
class SegmentPoolGradFunctor<platform::CUDADeviceContext, T, IndexT> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& out_grad,
                  const framework::Tensor& segments, framework::Tensor* in_grad,
                  const framework::Tensor* summed_ids = nullptr,
                  const std::string pooltype = "SUM") {
    if (pooltype == "MAX" || pooltype == "MIN") {
      SegmentPoolCUDAGradFunctor<T, IndexT>(context, input, segments, output,
                                            out_grad, in_grad, pooltype);
    } else if (pooltype == "MEAN") {
      framework::Tensor mean_grad;
      mean_grad.mutable_data<T>(input.dims(), context.GetPlace());
      framework::TensorCopy(out_grad, context.GetPlace(), context, &mean_grad);
      int len = output.dims()[0];
      int dim = output.numel() / len;
      auto config = platform::GetGpuLaunchConfig1D(context, len);
      SimpleDiv<T><<<config.block_per_grid.x, config.thread_per_block.x, 0,
                     context.stream()>>>(mean_grad.data<T>(),
                                         summed_ids->data<T>(), len, dim);
      GPUGather<T, IndexT>(context, mean_grad, segments, in_grad);
    } else if (pooltype == "SUM") {
      GPUGather<T, IndexT>(context, out_grad, segments, in_grad);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported segment pooling operation, Only MEAN, SUM, MAX, MIN "
          "available, but got %s.",
          pooltype));
    }
  }
};

using CUDA = paddle::platform::CUDADeviceContext;
template class SegmentPoolFunctor<CUDA, float, int>;
template class SegmentPoolFunctor<CUDA, float, int64_t>;
template class SegmentPoolFunctor<CUDA, double, int>;
template class SegmentPoolFunctor<CUDA, double, int64_t>;
template class SegmentPoolGradFunctor<CUDA, float, int>;
template class SegmentPoolGradFunctor<CUDA, float, int64_t>;
template class SegmentPoolGradFunctor<CUDA, double, int>;
template class SegmentPoolGradFunctor<CUDA, double, int64_t>;

}  // namespace operators
}  // namespace paddle
