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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace platform {
struct CUDAPlace;
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using DataLayout = platform::DataLayout;
using Tensor = framework::Tensor;

int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

static inline int SizeOutAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = axis + 1; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

template <typename T, int VLEN>
union vec_t {
  static_assert(sizeof(T) == -1, "vec_t is only available by specialization.");
};

template <>
union vec_t<float, 4> {
  float4 s;
  float v[4];
};

template <>
union vec_t<platform::float16, 4> {
  int2 s;
  platform::float16 v[4];
};

template <typename T, int WARP_BATCH, int WARP_SIZE_SOFTMAX>
__device__ __forceinline__ void warp_reduce_sum(T* sum) {
#pragma unroll
  for (int offset = WARP_SIZE_SOFTMAX / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      T sum_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = sum[i] + sum_val;
    }
  }
}

template <typename T, int WARP_BATCH, int WARP_SIZE_SOFTMAX>
__device__ __forceinline__ void warp_reduce_max(T* sum) {
#pragma unroll
  for (int offset = WARP_SIZE_SOFTMAX / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      T max_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = max(sum[i], max_val);
    }
  }
}

template <typename T, typename VECT, int VPT, int WARP_PER_BLOCK>
__global__ void VecSoftmaxForward(T* dst, const T* src, const int batch_size,
                                  const int softmax_ele) {
  int offset = blockIdx.x * softmax_ele * WARP_PER_BLOCK;
  int idx = threadIdx.x * VPT;

  VECT buf = reinterpret_cast<const VECT*>(&src[offset + idx])[0];
  T* bufp = reinterpret_cast<T*>(&buf);
  float4 val4;
  float* val4p = reinterpret_cast<float*>(&val4);
  for (int i = 0; i < VPT; ++i) {
    val4p[i] = static_cast<float>(bufp[i]);
  }
  float val = val4.x + val4.y + val4.z + val4.w;
  float max_val = math::warpReduceMax<float>(
      max(max(val4.x, val4.y), max(val4.z, val4.w)), 0xffffffff);
  float4 tmp4 = make_float4(__expf(val4.x - max_val), __expf(val4.y - max_val),
                            __expf(val4.z - max_val), __expf(val4.w - max_val));
  float* tmp4p = reinterpret_cast<float*>(&tmp4);
  float invsum = 1.f / (math::warpReduceSum<float>(
                            tmp4.x + tmp4.y + tmp4.z + tmp4.w, 0xffffffff) +
                        1e-6f);
  for (int i = 0; i < VPT; ++i) {
    bufp[i] = static_cast<T>(tmp4p[i] * invsum);
  }
  reinterpret_cast<VECT*>(&dst[offset + idx])[0] = buf;
}

template <typename T, typename acc_t, int log2_elements>
__global__ void WarpSoftmaxForward(T* dst, const T* src, const int batch_size,
                                   const int stride, const int element_count) {
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE_SOFTMAX =
      (next_power_of_two < 32) ? next_power_of_two : 32;
  constexpr int WARP_ITERATIONS_SOFTMAX = next_power_of_two / WARP_SIZE_SOFTMAX;
  constexpr int WARP_BATCH_SOFTMAX = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch =
      (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH_SOFTMAX;

  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH_SOFTMAX) {
    local_batches = WARP_BATCH_SOFTMAX;
  }

  int local_idx = threadIdx.x;

  src += first_batch * stride + local_idx;
  dst += first_batch * stride + local_idx;

  // load data from global memory
  acc_t elements[WARP_BATCH_SOFTMAX][WARP_ITERATIONS_SOFTMAX];
  for (int i = 0; i < WARP_BATCH_SOFTMAX; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS_SOFTMAX; ++it) {
      int element_index = local_idx + it * WARP_SIZE_SOFTMAX;
      if (element_index < batch_element_count) {
        elements[i][it] =
            static_cast<float>(src[i * element_count + it * WARP_SIZE_SOFTMAX]);
      } else {
        elements[i][it] = -std::numeric_limits<acc_t>::infinity();
      }
    }
  }

  // compute max_value
  acc_t max_value[WARP_BATCH_SOFTMAX];
#pragma unroll
  for (int i = 0; i < WARP_BATCH_SOFTMAX; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS_SOFTMAX; ++it) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  warp_reduce_max<acc_t, WARP_BATCH_SOFTMAX, WARP_SIZE_SOFTMAX>(max_value);

  acc_t sum[WARP_BATCH_SOFTMAX]{0.0f};
#pragma unroll
  for (int i = 0; i < WARP_BATCH_SOFTMAX; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS_SOFTMAX; ++it) {
      // elements[i][it] = static_cast<T>(
      //     std::exp(static_cast<float>(elements[i][it] - max_value[i])));
      elements[i][it] = (std::exp((elements[i][it] - max_value[i])));
      sum[i] += elements[i][it];
    }
  }
  warp_reduce_sum<acc_t, WARP_BATCH_SOFTMAX, WARP_SIZE_SOFTMAX>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH_SOFTMAX; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS_SOFTMAX; ++it) {
      int element_index = local_idx + it * WARP_SIZE_SOFTMAX;
      if (element_index < element_count) {
        dst[i * element_count + it * WARP_SIZE_SOFTMAX] =
            elements[i][it] / sum[i];
      } else {
        break;
      }
    }
  }
}

template <typename T>
__global__ void BlockSoftmaxForward(T* dst, const T* src) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float max_value = static_cast<float>(src[i]);
  max_value = math::blockReduceMax<float>(max_value, 0xffffffff);

  float sum_value = __expf(static_cast<float>(src[i]) - max_value);
  sum_value = math::blockReduceSum<float>(sum_value, 0xffffffff);

  dst[i] = static_cast<T>(__expf(static_cast<float>(src[i]) - max_value) /
                          (sum_value + 1e-6f));
}

template <typename T>
__global__ void SoftmaxForward(T* dst, const T* src, const int batch_size,
                               const int softmax_ele) {
  extern __shared__ float max_data[];
  extern __shared__ float sum_data[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  if (tid >= softmax_ele) {
    max_data[tid] = -std::numeric_limits<float>::infinity();
  } else {
    max_data[tid] = static_cast<float>(src[i]);
  }
  __syncthreads();
  // printf("a");

  for (unsigned int s = blockDim.x / 2; s > 16; s >>= 1) {
    if (tid < s) {
      max_data[tid] = max(max_data[tid], max_data[tid + s]);
    }
    __syncthreads();
  }
  // printf("b");

  float max_value;
  if (tid < 32) {
    max_value = max_data[tid];
    max_value = math::warpReduceMax<float>(max_value, 0xffffffff);
  }

  if (tid >= softmax_ele) {
    sum_data[tid] = 0;
  } else {
    sum_data[tid] = __expf(static_cast<float>(src[i]) - max_value);
  }
  __syncthreads();

  // printf("c");
  for (unsigned int s = blockDim.x / 2; s > 16; s >>= 1) {
    if (tid < s) {
      sum_data[tid] += sum_data[tid + s];
    }
    __syncthreads();
  }

  float sum_value;
  if (tid < 32) {
    sum_value = sum_data[tid];
    sum_value = math::warpReduceSum<float>(sum_value, 0xffffffff);
  }

  if (tid < softmax_ele) {
    dst[i] = static_cast<T>(__expf(static_cast<float>(src[i]) - max_value) /
                            (sum_value + 1e-6f));
  }
}

template <typename T, int VPT, int WARP_PER_BLOCK>
__global__ void VecSoftmaxBackward(T* dst, const T* grad, const T* src,
                                   const int batch_size,
                                   const int softmax_ele) {
  const int offset =
      blockIdx.x * softmax_ele * WARP_PER_BLOCK + threadIdx.x * VPT;

  float local_sum_gy = 0.f;
  vec_t<T, VPT> local_grad;
  vec_t<T, VPT> local_src;

  local_grad.s =
      reinterpret_cast<const decltype(local_grad.s)*>(&grad[offset])[0];
  local_src.s = reinterpret_cast<const decltype(local_src.s)*>(&src[offset])[0];

  for (int i = 0; i < VPT; ++i) {
    local_sum_gy += static_cast<float>(local_grad.v[i]) *
                    static_cast<float>(local_src.v[i]);
  }
  float sum_gy = math::warpReduceSum<float>(local_sum_gy, 0xffffffff);

  vec_t<T, VPT> local_dst;
  for (int i = 0; i < VPT; ++i) {
    local_dst.v[i] =
        static_cast<T>(static_cast<float>(local_src.v[i]) *
                       (static_cast<float>(local_grad.v[i]) - sum_gy));
  }
  reinterpret_cast<decltype(local_dst.s)*>(&dst[offset])[0] = local_dst.s;
}

template <typename T>
class SoftmaxCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    auto* out_data = out->data<T>();

    auto dims = x->dims();
    const int rank = dims.size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    const int dim = dims[axis];
    const int N = SizeToAxis(axis, dims);
    const int D = SizeOutAxis(axis, dims);

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    constexpr int max_grid_threads = 1024;
    bool optimize = false;
    constexpr int warps_per_block = 4;
    // if (D == 1 && dim <= max_grid_threads && sizeof(T) <= 4 && dim % 2 == 0)
    // {
    if (D == 1 && dim <= max_grid_threads && sizeof(T) <= 4) {
      if (dim == 128 && N % warps_per_block == 0) {
        // a warp for a batch, 4 elements for a thread, only support the softmax
        // dim size = 128 currently
        optimize = true;
        if (sizeof(T) == 2) {
          VecSoftmaxForward<T, int2, 4, warps_per_block><<<
              N / warps_per_block, warps_per_block * WARP_SIZE, 0,
              ctx.cuda_device_context().stream()>>>(out_data, x->data<T>(), N,
                                                    dim);
        } else if (sizeof(T) == 4) {
          VecSoftmaxForward<T, int4, 4, warps_per_block><<<
              N / warps_per_block, warps_per_block * WARP_SIZE, 0,
              ctx.cuda_device_context().stream()>>>(out_data, x->data<T>(), N,
                                                    dim);
        } else {
          assert(false && "not support");
        }
      } else if (dim < 32) {
        // LOG(INFO) << "N: " << N << "dim: " << dim;
        optimize = true;
        int log2_elements = static_cast<int>(log2_ceil(dim));
        // LOG(INFO)<<"log2_elements: "<< log2_elements;
        const int next_power_of_two = 1 << log2_elements;

        int warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (N + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);

        switch (log2_elements) {
#define LAUNCH_SOFTMAX_WARP_FORWARD(L2E)                                  \
  case L2E:                                                               \
    WarpSoftmaxForward<                                                   \
        T, float,                                                         \
        L2E><<<blocks, threads, 0, ctx.cuda_device_context().stream()>>>( \
        out_data, x->data<T>(), N, dim, dim);                             \
    break;

          LAUNCH_SOFTMAX_WARP_FORWARD(0);  // 1
          LAUNCH_SOFTMAX_WARP_FORWARD(1);  // 2
          LAUNCH_SOFTMAX_WARP_FORWARD(2);  // 4
          LAUNCH_SOFTMAX_WARP_FORWARD(3);  // 8
          LAUNCH_SOFTMAX_WARP_FORWARD(4);  // 16
          LAUNCH_SOFTMAX_WARP_FORWARD(5);  // 32
          default:
            break;
        }
      } else if (dim >= 32 && (dim % 32 == 0)) {
        optimize = true;
        BlockSoftmaxForward<
            T><<<N, dim, 0, ctx.cuda_device_context().stream()>>>(out_data,
                                                                  x->data<T>());
      } else if (dim >= 32 && dim <= 512) {
        optimize = true;
        // int warp_per_block = (dim + 32 - 1) / 32;
        // int block = warp_per_block * 32;
        // int grid = (N*dim + block - 1) / block;
        // LOG(INFO) << "N: " << N << "dim: " << dim;
        // LOG(INFO) << "block: " << block<<"grid: "<<grid;
        int block = dim;
        SoftmaxForward<T><<<N, block, block * sizeof(float),
                            ctx.cuda_device_context().stream()>>>(
            out_data, x->data<T>(), N, dim);
      }
    }
    if (!optimize) {
      ScopedTensorDescriptor desc;
      std::vector<int> tensor_dims = {N, dim, D, 1};
      DataLayout layout = DataLayout::kNCHW;
      cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);

      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto handle = dev_ctx.cudnn_handle();
      auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                   : CUDNN_SOFTMAX_MODE_CHANNEL;

      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxForward(
          handle, CUDNN_SOFTMAX_ACCURATE, mode,
          platform::CudnnDataType<T>::kOne(), desc_, x->data<T>(),
          platform::CudnnDataType<T>::kZero(), desc_, out_data));
    }
  }
};

template <typename T>
class SoftmaxGradCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());
    auto* dx_data = dx->data<T>();

    auto dims = out->dims();
    const int rank = dims.size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    const int dim = dims[axis];
    const int N = SizeToAxis(axis, dims);
    const int D = SizeOutAxis(axis, dims);

    constexpr int warps_per_block = 4;
    constexpr bool warp_softmax_available =
        std::is_same<T, float>::value ||
        std::is_same<T, platform::float16>::value;
    if (D == 1 && dim == 128 && N % warps_per_block == 0 &&
        warp_softmax_available) {
      if (std::is_same<T, float>::value) {
        VecSoftmaxBackward<
            float, 4,
            warps_per_block><<<N / warps_per_block, warps_per_block * WARP_SIZE,
                               0, ctx.cuda_device_context().stream()>>>(
            dx->data<float>(), dout->data<float>(), out->data<float>(), N, dim);
      } else if (std::is_same<T, platform::float16>::value) {
        VecSoftmaxBackward<
            platform::float16, 4,
            warps_per_block><<<N / warps_per_block, warps_per_block * WARP_SIZE,
                               0, ctx.cuda_device_context().stream()>>>(
            dx->data<platform::float16>(), dout->data<platform::float16>(),
            out->data<platform::float16>(), N, dim);
      } else {
        PADDLE_ENFORCE_EQ(
            warp_softmax_available, true,
            platform::errors::Unimplemented(
                "Warp softmax backward is only available for fp32 and fp16"));
      }
    } else {
      ScopedTensorDescriptor desc;
      std::vector<int> tensor_dims = {N, dim, D, 1};
      DataLayout layout = DataLayout::kNCHW;
      cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);

      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto handle = dev_ctx.cudnn_handle();
      auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                   : CUDNN_SOFTMAX_MODE_CHANNEL;

      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxBackward(
          handle, CUDNN_SOFTMAX_ACCURATE, mode,
          platform::CudnnDataType<T>::kOne(), desc_, out->data<T>(), desc_,
          dout->data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
          dx_data));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_KERNEL(softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float>,
                   ops::SoftmaxCUDNNKernel<double>,
                   ops::SoftmaxCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float>,
                   ops::SoftmaxGradCUDNNKernel<double>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16>);
