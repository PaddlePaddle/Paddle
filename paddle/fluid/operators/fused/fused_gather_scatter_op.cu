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

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/fused/fused_gather_scatter_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, typename IndexT>
struct GatherScatterSumCUDAFunctor {
  DEVICE inline void operator()(const T* params, T* output, const IndexT& in_i,
                                const IndexT& out_i) {
    paddle::platform::CudaAtomicAdd(output + out_i, *(params + in_i));
  }
};

template <typename T, typename IndexT>
struct GatherScatterMaxCUDAFunctor {
  DEVICE inline void operator()(const T* params, T* output, const IndexT& in_i,
                                const IndexT& out_i) {
    paddle::platform::CudaAtomicMax(output + out_i, *(params + in_i));
  }
};

template <typename T, typename IndexT>
struct GatherScatterMinCUDAFunctor {
  DEVICE inline void operator()(const T* params, T* output, const IndexT& in_i,
                                const IndexT& out_i) {
    paddle::platform::CudaAtomicMin(output + out_i, *(params + in_i));
  }
};

template <typename T, typename IndexT, typename Functor>
__global__ void GatherScatterCUDAKernel(const T* params,
                                        const IndexT* gather_indices,
                                        const IndexT* scatter_indices,
                                        T* output, size_t index_size,
                                        size_t slice_size, Functor functor) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;
    IndexT gather_i = gather_indices[indices_i];
    IndexT scatter_i = scatter_indices[indices_i];
    int64_t in_i = gather_i * slice_size + slice_i;
    int64_t out_i = scatter_i * slice_size + slice_i;
    functor(params, output, in_i, out_i);
  }
}

// For min and max
template <typename T>
__global__ void InputResetCUDAKernel(T* output, size_t input_size,
                                     size_t slice_size) {
  CUDA_KERNEL_LOOP_TYPE(i, input_size * slice_size, int64_t) {
    if (*(output + i) == std::numeric_limits<T>::min() ||
        *(output + i) == std::numeric_limits<T>::max()) {
      *(output + i) = 0;
    }
  }
}

// Get scatter_count
template <typename T, typename IndexT>
__global__ void ComputeCountCUDAKernel(int* count,
                                       const IndexT* scatter_indices,
                                       size_t index_size) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size, int64_t) {
    IndexT scatter_i = scatter_indices[i];
    paddle::platform::CudaAtomicAdd(count + scatter_i, 1);
  }
}

// For forward mean
template <typename T>
__global__ void ManipulateMeanCUDAKernel(T* output, int* count,
                                         size_t input_size, size_t slice_size) {
  CUDA_KERNEL_LOOP_TYPE(i, input_size * slice_size, int64_t) {
    int64_t c_index = i / slice_size;
    if (*(count + c_index) > 1) {
      *(output + i) = *(output + i) / *(count + c_index);
    }
  }
}

// For backward mean
template <typename T, typename IndexT>
__global__ void ManipulateMeanGradCUDAKernel(const T* params,
                                             const IndexT* gather_indices,
                                             const IndexT* scatter_indices,
                                             T* output, size_t index_size,
                                             size_t slice_size,
                                             const int* scatter_count) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;
    IndexT gather_i = gather_indices[indices_i];
    IndexT scatter_i = scatter_indices[indices_i];
    int64_t in_i = gather_i * slice_size + slice_i;
    int64_t out_i = scatter_i * slice_size + slice_i;
    paddle::platform::CudaAtomicAdd(output + out_i,
                                    *(params + in_i) / scatter_count[gather_i]);
  }
}

template <typename DeviceContext, typename T, typename IndexT>
class FusedGatherScatterOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<Tensor>("X");
    auto* gather_index = ctx.Input<Tensor>("Gather_index");
    auto* scatter_index = ctx.Input<Tensor>("Scatter_index");
    auto* Y = ctx.Output<Tensor>("Out");
    std::string pool_type = ctx.Attr<std::string>("pool_type");

    const int& index_size = gather_index->dims()[0];
    if (index_size == 0) return;

    T* p_output = Y->mutable_data<T>(ctx.GetPlace());
    const auto& src_dims = X->dims();
    int64_t memset_size = 1;
    for (int i = 0; i < src_dims.size(); ++i) {
      memset_size *= src_dims[i];
    }
    const size_t& memset_bytes = memset_size * sizeof(T);
    if (pool_type == "SUM" || pool_type == "MEAN") {
      cudaMemset(p_output, 0, memset_bytes);
    } else if (pool_type == "MAX") {
      thrust::device_ptr<T> p_output_ptr(p_output);
      thrust::fill(thrust::device, p_output_ptr, p_output_ptr + memset_size,
                   std::numeric_limits<T>::min());
    } else if (pool_type == "MIN") {
      thrust::device_ptr<T> p_output_ptr(p_output);
      thrust::fill(thrust::device, p_output_ptr, p_output_ptr + memset_size,
                   std::numeric_limits<T>::max());
    }

    int64_t slice_size = 1;
    for (int i = 1; i < src_dims.size(); ++i) {
      slice_size *= src_dims[i];
    }
    const T* p_src = X->data<T>();
    const IndexT* g_index = gather_index->data<IndexT>();
    const IndexT* s_index = scatter_index->data<IndexT>();

    int block = 512;
    int64_t n = slice_size * index_size;
    int64_t grid = (n + block - 1) / block;
    int64_t input_size = src_dims[0];
    if (pool_type == "SUM") {
      GatherScatterSumCUDAFunctor<T, IndexT> functor;
      GatherScatterCUDAKernel<T, IndexT,
                              GatherScatterSumCUDAFunctor<T, IndexT>><<<
          grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                              ctx.device_context())
                              .stream()>>>(p_src, g_index, s_index, p_output,
                                           index_size, slice_size, functor);
    } else if (pool_type == "MAX") {
      GatherScatterMaxCUDAFunctor<T, IndexT> functor;
      GatherScatterCUDAKernel<T, IndexT,
                              GatherScatterMaxCUDAFunctor<T, IndexT>><<<
          grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                              ctx.device_context())
                              .stream()>>>(p_src, g_index, s_index, p_output,
                                           index_size, slice_size, functor);

      int64_t grid_max = (input_size * slice_size + block - 1) / block;
      InputResetCUDAKernel<
          T><<<grid_max, block, 0,
               reinterpret_cast<const platform::CUDADeviceContext&>(
                   ctx.device_context())
                   .stream()>>>(p_output, input_size, slice_size);
    } else if (pool_type == "MIN") {
      GatherScatterMinCUDAFunctor<T, IndexT> functor;
      GatherScatterCUDAKernel<T, IndexT,
                              GatherScatterMinCUDAFunctor<T, IndexT>><<<
          grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                              ctx.device_context())
                              .stream()>>>(p_src, g_index, s_index, p_output,
                                           index_size, slice_size, functor);

      int64_t grid_min = (input_size * slice_size + block - 1) / block;
      InputResetCUDAKernel<
          T><<<grid_min, block, 0,
               reinterpret_cast<const platform::CUDADeviceContext&>(
                   ctx.device_context())
                   .stream()>>>(p_output, input_size, slice_size);
    } else if (pool_type == "MEAN") {
      GatherScatterSumCUDAFunctor<T, IndexT> functor;
      GatherScatterCUDAKernel<T, IndexT,
                              GatherScatterSumCUDAFunctor<T, IndexT>><<<
          grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                              ctx.device_context())
                              .stream()>>>(p_src, g_index, s_index, p_output,
                                           index_size, slice_size, functor);

      auto* scatter_count = ctx.Output<Tensor>("Scatter_count");
      int* p_scatter_count = scatter_count->mutable_data<int>(ctx.GetPlace());
      cudaMemset(p_scatter_count, 0, input_size * sizeof(int));
      int64_t grid_count = (index_size + block - 1) / block;
      ComputeCountCUDAKernel<
          T, IndexT><<<grid_count, block, 0,
                       reinterpret_cast<const platform::CUDADeviceContext&>(
                           ctx.device_context())
                           .stream()>>>(p_scatter_count, s_index, index_size);

      int64_t grid_mean = (input_size * slice_size + block - 1) / block;
      ManipulateMeanCUDAKernel<
          T><<<grid_mean, block, 0,
               reinterpret_cast<const platform::CUDADeviceContext&>(
                   ctx.device_context())
                   .stream()>>>(p_output, p_scatter_count, input_size,
                                slice_size);
    }
  }
};

template <typename DeviceContext, typename T, typename IndexT>
class FusedGatherScatterGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* gather_index = ctx.Input<Tensor>("Scatter_index");
    auto* scatter_index = ctx.Input<Tensor>("Gather_index");
    auto* Y = ctx.Output<Tensor>(framework::GradVarName("X"));
    std::string pool_type = ctx.Attr<std::string>("pool_type");

    const int& index_size = gather_index->dims()[0];
    if (index_size == 0) return;

    T* p_output = Y->mutable_data<T>(ctx.GetPlace());
    const auto& src_dims = X->dims();
    int64_t memset_size = 1;
    for (int i = 0; i < src_dims.size(); ++i) {
      memset_size *= src_dims[i];
    }
    const size_t& memset_bytes = memset_size * sizeof(T);
    if (pool_type == "SUM" || pool_type == "MEAN") {
      cudaMemset(p_output, 0, memset_bytes);
    } else if (pool_type == "MAX") {
      thrust::device_ptr<T> p_output_ptr(p_output);
      thrust::fill(thrust::device, p_output_ptr, p_output_ptr + memset_size,
                   std::numeric_limits<T>::min());
    } else if (pool_type == "MIN") {
      thrust::device_ptr<T> p_output_ptr(p_output);
      thrust::fill(thrust::device, p_output_ptr, p_output_ptr + memset_size,
                   std::numeric_limits<T>::max());
    }

    int64_t slice_size = 1;
    for (int i = 1; i < src_dims.size(); ++i) {
      slice_size *= src_dims[i];
    }
    const T* p_src = X->data<T>();
    const IndexT* g_index = gather_index->data<IndexT>();
    const IndexT* s_index = scatter_index->data<IndexT>();

    int block = 512;
    int64_t n = slice_size * index_size;
    int64_t grid = (n + block - 1) / block;
    int64_t input_size = src_dims[0];
    if (pool_type == "SUM") {
      GatherScatterSumCUDAFunctor<T, IndexT> functor;
      GatherScatterCUDAKernel<T, IndexT,
                              GatherScatterSumCUDAFunctor<T, IndexT>><<<
          grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                              ctx.device_context())
                              .stream()>>>(p_src, g_index, s_index, p_output,
                                           index_size, slice_size, functor);
    } else if (pool_type == "MAX") {
      GatherScatterMaxCUDAFunctor<T, IndexT> functor;
      GatherScatterCUDAKernel<T, IndexT,
                              GatherScatterMaxCUDAFunctor<T, IndexT>><<<
          grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                              ctx.device_context())
                              .stream()>>>(p_src, g_index, s_index, p_output,
                                           index_size, slice_size, functor);

      int64_t grid_max = (input_size * slice_size + block - 1) / block;
      InputResetCUDAKernel<
          T><<<grid_max, block, 0,
               reinterpret_cast<const platform::CUDADeviceContext&>(
                   ctx.device_context())
                   .stream()>>>(p_output, input_size, slice_size);
    } else if (pool_type == "MIN") {
      GatherScatterMinCUDAFunctor<T, IndexT> functor;
      GatherScatterCUDAKernel<T, IndexT,
                              GatherScatterMinCUDAFunctor<T, IndexT>><<<
          grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                              ctx.device_context())
                              .stream()>>>(p_src, g_index, s_index, p_output,
                                           index_size, slice_size, functor);

      int64_t grid_min = (input_size * slice_size + block - 1) / block;
      InputResetCUDAKernel<
          T><<<grid_min, block, 0,
               reinterpret_cast<const platform::CUDADeviceContext&>(
                   ctx.device_context())
                   .stream()>>>(p_output, input_size, slice_size);
    } else if (pool_type == "MEAN") {
      auto* scatter_count = ctx.Input<Tensor>("Scatter_count");
      const int* s_count = scatter_count->data<int>();
      ManipulateMeanGradCUDAKernel<T, IndexT><<<
          grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                              ctx.device_context())
                              .stream()>>>(p_src, g_index, s_index, p_output,
                                           index_size, slice_size, s_count);
    }
  }
};

}  // namespace operators
}  // namespace paddle

using CUDA = paddle::platform::CUDADeviceContext;
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    fused_gather_scatter, ops::FusedGatherScatterOpCUDAKernel<CUDA, float, int>,
    ops::FusedGatherScatterOpCUDAKernel<CUDA, float, int64_t>,
    ops::FusedGatherScatterOpCUDAKernel<CUDA, double, int>,
    ops::FusedGatherScatterOpCUDAKernel<CUDA, double, int64_t>,
    ops::FusedGatherScatterOpCUDAKernel<CUDA, int, int>,
    ops::FusedGatherScatterOpCUDAKernel<CUDA, int, int64_t>,
    ops::FusedGatherScatterOpCUDAKernel<CUDA, int64_t, int>,
    ops::FusedGatherScatterOpCUDAKernel<CUDA, int64_t, int64_t>);

REGISTER_OP_CUDA_KERNEL(
    fused_gather_scatter_grad,
    ops::FusedGatherScatterGradOpCUDAKernel<CUDA, float, int>,
    ops::FusedGatherScatterGradOpCUDAKernel<CUDA, float, int64_t>,
    ops::FusedGatherScatterGradOpCUDAKernel<CUDA, double, int>,
    ops::FusedGatherScatterGradOpCUDAKernel<CUDA, double, int64_t>,
    ops::FusedGatherScatterGradOpCUDAKernel<CUDA, int, int>,
    ops::FusedGatherScatterGradOpCUDAKernel<CUDA, int, int64_t>,
    ops::FusedGatherScatterGradOpCUDAKernel<CUDA, int64_t, int>,
    ops::FusedGatherScatterGradOpCUDAKernel<CUDA, int64_t, int64_t>);
