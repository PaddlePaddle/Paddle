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
#include "paddle/fluid/operators/graph_send_recv_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, typename IndexT>
struct GraphSendRecvSumCUDAFunctor {
  DEVICE inline void operator()(const T* params, T* output, const IndexT& in_i,
                                const IndexT& out_i) {
    paddle::platform::CudaAtomicAdd(output + out_i, *(params + in_i));
  }
};

template <typename T, typename IndexT>
struct GraphSendRecvMaxCUDAFunctor {
  DEVICE inline void operator()(const T* params, T* output, const IndexT& in_i,
                                const IndexT& out_i) {
    paddle::platform::CudaAtomicMax(output + out_i, *(params + in_i));
  }
};

template <typename T, typename IndexT>
struct GraphSendRecvMinCUDAFunctor {
  DEVICE inline void operator()(const T* params, T* output, const IndexT& in_i,
                                const IndexT& out_i) {
    paddle::platform::CudaAtomicMin(output + out_i, *(params + in_i));
  }
};

template <typename T, typename IndexT, typename Functor>
__global__ void GraphSendRecvCUDAKernel(const T* params,
                                        const IndexT* src_indices,
                                        const IndexT* dst_indices, T* output,
                                        size_t index_size, size_t slice_size,
                                        Functor functor) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;
    IndexT src_i = src_indices[indices_i];
    IndexT dst_i = dst_indices[indices_i];
    int64_t in_i = src_i * slice_size + slice_i;
    int64_t out_i = dst_i * slice_size + slice_i;
    functor(params, output, in_i, out_i);
  }
}

// For max
template <typename T>
__global__ void InputResetMaxCUDAKernel(T* output, size_t input_size,
                                        size_t slice_size) {
  CUDA_KERNEL_LOOP_TYPE(i, input_size * slice_size, int64_t) {
    if (*(output + i) == std::numeric_limits<T>::min()) {
      *(output + i) = 0;
    }
  }
}

// For min
template <typename T>
__global__ void InputResetMinCUDAKernel(T* output, size_t input_size,
                                        size_t slice_size) {
  CUDA_KERNEL_LOOP_TYPE(i, input_size * slice_size, int64_t) {
    if (*(output + i) == std::numeric_limits<T>::max()) {
      *(output + i) = 0;
    }
  }
}

// Get dst_count
template <typename T, typename IndexT>
__global__ void ComputeCountCUDAKernel(int* count, const IndexT* dst_indices,
                                       size_t index_size) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size, int64_t) {
    IndexT dst_i = dst_indices[i];
    paddle::platform::CudaAtomicAdd(count + dst_i, 1);
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
__global__ void ManipulateMeanGradCUDAKernel(
    const T* params, const IndexT* src_indices, const IndexT* dst_indices,
    T* output, size_t index_size, size_t slice_size, const int* dst_count) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;
    IndexT src_i = src_indices[indices_i];
    IndexT dst_i = dst_indices[indices_i];
    int64_t in_i = src_i * slice_size + slice_i;
    int64_t out_i = dst_i * slice_size + slice_i;
    paddle::platform::CudaAtomicAdd(output + out_i,
                                    *(params + in_i) / dst_count[src_i]);
  }
}

// For backward min and max
template <typename T, typename IndexT>
__global__ void ManipulateMinMaxGradCUDAKernel(
    const T* params, const IndexT* src_indices, const IndexT* dst_indices,
    T* output, size_t index_size, size_t slice_size, const T* ptr_input,
    const T* ptr_output) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;
    IndexT src_i = src_indices[indices_i];
    IndexT dst_i = dst_indices[indices_i];
    int64_t in_i = src_i * slice_size + slice_i;
    int64_t out_i = dst_i * slice_size + slice_i;
    paddle::platform::CudaAtomicAdd(
        output + out_i,
        *(params + in_i) * (*(ptr_input + out_i) == *(ptr_output + in_i)));
  }
}

template <typename DeviceContext, typename T, typename IndexT>
void GraphSendRecvOpCUDAKernelLaunchHelper(
    const framework::ExecutionContext& ctx, const Tensor& src_index,
    const Tensor& dst_index) {
  auto* X = ctx.Input<Tensor>("X");
  auto* Y = ctx.Output<Tensor>("Out");
  std::string pool_type = ctx.Attr<std::string>("pool_type");

  const int& index_size = src_index.dims()[0];

  T* p_output = Y->mutable_data<T>(ctx.GetPlace());
  const auto& src_dims = X->dims();
  int64_t memset_size = 1;
  for (int i = 0; i < src_dims.size(); ++i) {
    memset_size *= src_dims[i];
  }
  const size_t& memset_bytes = memset_size * sizeof(T);
  if (pool_type == "SUM" || pool_type == "MEAN") {
#ifdef PADDLE_WITH_HIP
    hipMemset(p_output, 0, memset_bytes);
#else
    cudaMemset(p_output, 0, memset_bytes);
#endif
  } else if (pool_type == "MAX") {
    thrust::device_ptr<T> p_output_ptr(p_output);
    thrust::fill(thrust::device, p_output_ptr, p_output_ptr + memset_size,
                 std::numeric_limits<T>::min());
  } else if (pool_type == "MIN") {
    thrust::device_ptr<T> p_output_ptr(p_output);
    thrust::fill(thrust::device, p_output_ptr, p_output_ptr + memset_size,
                 std::numeric_limits<T>::max());
  }

  if (index_size == 0) return;

  int64_t slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) {
    slice_size *= src_dims[i];
  }
  const T* p_src = X->data<T>();
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();

#ifdef PADDLE_WITH_HIP
  int block = 256;
#else
  int block = 1024;
#endif
  int64_t n = slice_size * index_size;
  const auto& dev_ctx = ctx.cuda_device_context();
  int64_t max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int64_t grid_tmp = (n + block - 1) / block;
  int64_t grid = grid_tmp < max_grid_dimx ? grid_tmp : max_grid_dimx;
  int64_t input_size = src_dims[0];
  if (pool_type == "SUM") {
    GraphSendRecvSumCUDAFunctor<T, IndexT> functor;
    GraphSendRecvCUDAKernel<T, IndexT,
                            GraphSendRecvSumCUDAFunctor<T, IndexT>><<<
        grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                            ctx.device_context())
                            .stream()>>>(p_src, s_index, d_index, p_output,
                                         index_size, slice_size, functor);
  } else if (pool_type == "MAX") {
    GraphSendRecvMaxCUDAFunctor<T, IndexT> functor;
    GraphSendRecvCUDAKernel<T, IndexT,
                            GraphSendRecvMaxCUDAFunctor<T, IndexT>><<<
        grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                            ctx.device_context())
                            .stream()>>>(p_src, s_index, d_index, p_output,
                                         index_size, slice_size, functor);

    int64_t grid_max_tmp = (input_size * slice_size + block - 1) / block;
    int64_t grid_max =
        grid_max_tmp < max_grid_dimx ? grid_max_tmp : max_grid_dimx;
    InputResetMaxCUDAKernel<
        T><<<grid_max, block, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(
                 ctx.device_context())
                 .stream()>>>(p_output, input_size, slice_size);
  } else if (pool_type == "MIN") {
    GraphSendRecvMinCUDAFunctor<T, IndexT> functor;
    GraphSendRecvCUDAKernel<T, IndexT,
                            GraphSendRecvMinCUDAFunctor<T, IndexT>><<<
        grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                            ctx.device_context())
                            .stream()>>>(p_src, s_index, d_index, p_output,
                                         index_size, slice_size, functor);

    int64_t grid_min_tmp = (input_size * slice_size + block - 1) / block;
    int64_t grid_min =
        grid_min_tmp < max_grid_dimx ? grid_min_tmp : max_grid_dimx;
    InputResetMinCUDAKernel<
        T><<<grid_min, block, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(
                 ctx.device_context())
                 .stream()>>>(p_output, input_size, slice_size);
  } else if (pool_type == "MEAN") {
    GraphSendRecvSumCUDAFunctor<T, IndexT> functor;
    GraphSendRecvCUDAKernel<T, IndexT,
                            GraphSendRecvSumCUDAFunctor<T, IndexT>><<<
        grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                            ctx.device_context())
                            .stream()>>>(p_src, s_index, d_index, p_output,
                                         index_size, slice_size, functor);

    auto* dst_count = ctx.Output<Tensor>("Dst_count");
    int* p_dst_count = dst_count->mutable_data<int>(ctx.GetPlace());

#ifdef PADDLE_WITH_HIP
    hipMemset(p_dst_count, 0, input_size * sizeof(int));
#else
    cudaMemset(p_dst_count, 0, input_size * sizeof(int));
#endif

    int64_t grid_count = (index_size + block - 1) / block;
    ComputeCountCUDAKernel<
        T, IndexT><<<grid_count, block, 0,
                     reinterpret_cast<const platform::CUDADeviceContext&>(
                         ctx.device_context())
                         .stream()>>>(p_dst_count, d_index, index_size);

    int64_t grid_mean_tmp = (input_size * slice_size + block - 1) / block;
    int64_t grid_mean =
        grid_mean_tmp < max_grid_dimx ? grid_mean_tmp : max_grid_dimx;
    ManipulateMeanCUDAKernel<
        T><<<grid_mean, block, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(
                 ctx.device_context())
                 .stream()>>>(p_output, p_dst_count, input_size, slice_size);
  }
}

template <typename DeviceContext, typename T, typename IndexT>
void GraphSendRecvGradOpCUDAKernelLaunchHelper(
    const framework::ExecutionContext& ctx, const Tensor& src_index,
    const Tensor& dst_index) {
  auto* X = ctx.Input<Tensor>(framework::GradVarName("Out"));
  auto* Y = ctx.Output<Tensor>(framework::GradVarName("X"));
  std::string pool_type = ctx.Attr<std::string>("pool_type");

  const int& index_size = src_index.dims()[0];

  T* p_output = Y->mutable_data<T>(ctx.GetPlace());
  const auto& src_dims = X->dims();
  int64_t memset_size = 1;
  for (int i = 0; i < src_dims.size(); ++i) {
    memset_size *= src_dims[i];
  }
  const size_t& memset_bytes = memset_size * sizeof(T);

#ifdef PADDLE_WITH_HIP
  hipMemset(p_output, 0, memset_bytes);
#else
  cudaMemset(p_output, 0, memset_bytes);
#endif

  if (index_size == 0) return;

  int64_t slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) {
    slice_size *= src_dims[i];
  }
  const T* p_src = X->data<T>();
  const IndexT* s_index = src_index.data<IndexT>();
  const IndexT* d_index = dst_index.data<IndexT>();

#ifdef PADDLE_WITH_HIP
  int block = 256;
#else
  int block = 1024;
#endif
  int64_t n = slice_size * index_size;
  const auto& dev_ctx = ctx.cuda_device_context();
  int64_t max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int64_t grid_tmp = (n + block - 1) / block;
  int64_t grid = grid_tmp < max_grid_dimx ? grid_tmp : max_grid_dimx;
  int64_t input_size = src_dims[0];
  if (pool_type == "SUM") {
    GraphSendRecvSumCUDAFunctor<T, IndexT> functor;
    GraphSendRecvCUDAKernel<T, IndexT,
                            GraphSendRecvSumCUDAFunctor<T, IndexT>><<<
        grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                            ctx.device_context())
                            .stream()>>>(p_src, s_index, d_index, p_output,
                                         index_size, slice_size, functor);
  } else if (pool_type == "MEAN") {
    auto* dst_count = ctx.Input<Tensor>("Dst_count");
    const int* s_count = dst_count->data<int>();
    ManipulateMeanGradCUDAKernel<T, IndexT><<<
        grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                            ctx.device_context())
                            .stream()>>>(p_src, s_index, d_index, p_output,
                                         index_size, slice_size, s_count);
  } else if (pool_type == "MAX" || pool_type == "MIN") {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Input<Tensor>("Out");
    const T* ptr_input = input->data<T>();
    const T* ptr_output = output->data<T>();
    ManipulateMinMaxGradCUDAKernel<T, IndexT><<<
        grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                            ctx.device_context())
                            .stream()>>>(p_src, s_index, d_index, p_output,
                                         index_size, slice_size, ptr_input,
                                         ptr_output);
  }
}

template <typename DeviceContext, typename T>
class GraphSendRecvOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* src_index = ctx.Input<Tensor>("Src_index");
    auto* dst_index = ctx.Input<Tensor>("Dst_index");
    auto index_type = framework::TransToProtoVarType(src_index->dtype());

    if (index_type == framework::proto::VarType::INT32) {
      GraphSendRecvOpCUDAKernelLaunchHelper<DeviceContext, T, int>(
          ctx, *src_index, *dst_index);
    } else if (index_type == framework::proto::VarType::INT64) {
      GraphSendRecvOpCUDAKernelLaunchHelper<DeviceContext, T, int64_t>(
          ctx, *src_index, *dst_index);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported Src_index or Dst_index dtype, expected int, int64, but "
          "got %s.",
          index_type));
    }
  }
};

template <typename DeviceContext, typename T>
class GraphSendRecvGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* src_index = ctx.Input<Tensor>("Dst_index");
    auto* dst_index = ctx.Input<Tensor>("Src_index");
    auto index_type = framework::TransToProtoVarType(src_index->dtype());

    if (index_type == framework::proto::VarType::INT32) {
      GraphSendRecvGradOpCUDAKernelLaunchHelper<DeviceContext, T, int>(
          ctx, *src_index, *dst_index);
    } else if (index_type == framework::proto::VarType::INT64) {
      GraphSendRecvGradOpCUDAKernelLaunchHelper<DeviceContext, T, int64_t>(
          ctx, *src_index, *dst_index);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported Src_index or Dst_index dtype, expected int, int64, but "
          "got %s.",
          index_type));
    }
  }
};

}  // namespace operators
}  // namespace paddle

using CUDA = paddle::platform::CUDADeviceContext;
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(graph_send_recv,
                        ops::GraphSendRecvOpCUDAKernel<CUDA, float>,
                        ops::GraphSendRecvOpCUDAKernel<CUDA, double>,
                        ops::GraphSendRecvOpCUDAKernel<CUDA, int>,
                        ops::GraphSendRecvOpCUDAKernel<CUDA, int64_t>);

REGISTER_OP_CUDA_KERNEL(graph_send_recv_grad,
                        ops::GraphSendRecvGradOpCUDAKernel<CUDA, float>,
                        ops::GraphSendRecvGradOpCUDAKernel<CUDA, double>,
                        ops::GraphSendRecvGradOpCUDAKernel<CUDA, int>,
                        ops::GraphSendRecvGradOpCUDAKernel<CUDA, int64_t>);
