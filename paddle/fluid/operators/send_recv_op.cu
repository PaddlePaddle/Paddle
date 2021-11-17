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
#include "paddle/fluid/operators/send_recv_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, typename IndexT>
struct SendRecvSumCUDAFunctor {
  DEVICE inline void operator()(const T* params, T* output, const IndexT& in_i,
                                const IndexT& out_i) {
    paddle::platform::CudaAtomicAdd(output + out_i, *(params + in_i));
  }
};

template <typename T, typename IndexT>
struct SendRecvMaxCUDAFunctor {
  DEVICE inline void operator()(const T* params, T* output, const IndexT& in_i,
                                const IndexT& out_i) {
    paddle::platform::CudaAtomicMax(output + out_i, *(params + in_i));
  }
};

template <typename T, typename IndexT>
struct SendRecvMinCUDAFunctor {
  DEVICE inline void operator()(const T* params, T* output, const IndexT& in_i,
                                const IndexT& out_i) {
    paddle::platform::CudaAtomicMin(output + out_i, *(params + in_i));
  }
};

template <typename T, typename IndexT, typename Functor>
__global__ void SendRecvCUDAKernel(const T* params, const IndexT* src_indices,
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
void SendRecvOpCUDAKernelLaunchHelper(const framework::ExecutionContext& ctx) {
  auto* X = ctx.Input<Tensor>("X");
  auto* src_index = ctx.Input<Tensor>("Src_index");
  auto* dst_index = ctx.Input<Tensor>("Dst_index");
  auto* Y = ctx.Output<Tensor>("Out");
  std::string pool_type = ctx.Attr<std::string>("pool_type");

  const int& index_size = src_index->dims()[0];

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
  const IndexT* s_index = src_index->data<IndexT>();
  const IndexT* d_index = dst_index->data<IndexT>();

  int block = 512;
  int64_t n = slice_size * index_size;
  int64_t grid = (n + block - 1) / block;
  int64_t input_size = src_dims[0];
  if (pool_type == "SUM") {
    SendRecvSumCUDAFunctor<T, IndexT> functor;
    SendRecvCUDAKernel<T, IndexT, SendRecvSumCUDAFunctor<T, IndexT>><<<
        grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                            ctx.device_context())
                            .stream()>>>(p_src, s_index, d_index, p_output,
                                         index_size, slice_size, functor);
  } else if (pool_type == "MAX") {
    SendRecvMaxCUDAFunctor<T, IndexT> functor;
    SendRecvCUDAKernel<T, IndexT, SendRecvMaxCUDAFunctor<T, IndexT>><<<
        grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                            ctx.device_context())
                            .stream()>>>(p_src, s_index, d_index, p_output,
                                         index_size, slice_size, functor);

    int64_t grid_max = (input_size * slice_size + block - 1) / block;
    InputResetCUDAKernel<
        T><<<grid_max, block, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(
                 ctx.device_context())
                 .stream()>>>(p_output, input_size, slice_size);
  } else if (pool_type == "MIN") {
    SendRecvMinCUDAFunctor<T, IndexT> functor;
    SendRecvCUDAKernel<T, IndexT, SendRecvMinCUDAFunctor<T, IndexT>><<<
        grid, block, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                            ctx.device_context())
                            .stream()>>>(p_src, s_index, d_index, p_output,
                                         index_size, slice_size, functor);

    int64_t grid_min = (input_size * slice_size + block - 1) / block;
    InputResetCUDAKernel<
        T><<<grid_min, block, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(
                 ctx.device_context())
                 .stream()>>>(p_output, input_size, slice_size);
  } else if (pool_type == "MEAN") {
    SendRecvSumCUDAFunctor<T, IndexT> functor;
    SendRecvCUDAKernel<T, IndexT, SendRecvSumCUDAFunctor<T, IndexT>><<<
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

    int64_t grid_mean = (input_size * slice_size + block - 1) / block;
    ManipulateMeanCUDAKernel<
        T><<<grid_mean, block, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(
                 ctx.device_context())
                 .stream()>>>(p_output, p_dst_count, input_size, slice_size);
  }
}

template <typename DeviceContext, typename T, typename IndexT>
void SendRecvGradOpCUDAKernelLaunchHelper(
    const framework::ExecutionContext& ctx) {
  auto* X = ctx.Input<Tensor>(framework::GradVarName("Out"));
  auto* src_index = ctx.Input<Tensor>("Dst_index");
  auto* dst_index = ctx.Input<Tensor>("Src_index");
  auto* Y = ctx.Output<Tensor>(framework::GradVarName("X"));
  std::string pool_type = ctx.Attr<std::string>("pool_type");

  const int& index_size = src_index->dims()[0];
  if (index_size == 0) return;

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

  int64_t slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) {
    slice_size *= src_dims[i];
  }
  const T* p_src = X->data<T>();
  const IndexT* s_index = src_index->data<IndexT>();
  const IndexT* d_index = dst_index->data<IndexT>();

  int block = 512;
  int64_t n = slice_size * index_size;
  int64_t grid = (n + block - 1) / block;
  int64_t input_size = src_dims[0];

  if (pool_type == "SUM") {
    SendRecvSumCUDAFunctor<T, IndexT> functor;
    SendRecvCUDAKernel<T, IndexT, SendRecvSumCUDAFunctor<T, IndexT>><<<
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
class SendRecvOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* src_index = ctx.Input<Tensor>("Src_index");
    auto index_type = src_index->type();

    if (index_type == framework::proto::VarType::INT32) {
      SendRecvOpCUDAKernelLaunchHelper<DeviceContext, T, int>(ctx);
    } else if (index_type == framework::proto::VarType::INT64) {
      SendRecvOpCUDAKernelLaunchHelper<DeviceContext, T, int64_t>(ctx);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported Src_index or Dst_index type, Expected int, int64, but "
          "got %s.",
          index_type));
    }
  }
};

template <typename DeviceContext, typename T>
class SendRecvGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* src_index = ctx.Input<Tensor>("Dst_index");
    auto index_type = src_index->type();
    if (index_type == framework::proto::VarType::INT32) {
      SendRecvGradOpCUDAKernelLaunchHelper<DeviceContext, T, int>(ctx);
    } else if (index_type == framework::proto::VarType::INT64) {
      SendRecvGradOpCUDAKernelLaunchHelper<DeviceContext, T, int64_t>(ctx);
    }
  }
};

}  // namespace operators
}  // namespace paddle

using CUDA = paddle::platform::CUDADeviceContext;
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(send_recv, ops::SendRecvOpCUDAKernel<CUDA, float>,
                        ops::SendRecvOpCUDAKernel<CUDA, double>,
                        ops::SendRecvOpCUDAKernel<CUDA, int>,
                        ops::SendRecvOpCUDAKernel<CUDA, int64_t>);

REGISTER_OP_CUDA_KERNEL(send_recv_grad,
                        ops::SendRecvGradOpCUDAKernel<CUDA, float>,
                        ops::SendRecvGradOpCUDAKernel<CUDA, double>,
                        ops::SendRecvGradOpCUDAKernel<CUDA, int>,
                        ops::SendRecvGradOpCUDAKernel<CUDA, int64_t>);
