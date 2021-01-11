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
#include "paddle/fluid/platform/gpu_launch_config.h"

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

#define LAUNCH_SOFTMAX_WARP_FORWARD(Log2Elements)                  \
  case Log2Elements:                                               \
    WarpSoftmaxForward<T, float, Log2Elements><<<                  \
        blocks, threads, 0, ctx.cuda_device_context().stream()>>>( \
        out_data, x->data<T>(), N, dim, dim);                      \
    break;

#define LAUNCH_SOFTMAX_WARP_BACKWARD(Log2Elements)                 \
  case Log2Elements:                                               \
    softmax_warp_backward<T, float, Log2Elements><<<               \
        blocks, threads, 0, ctx.cuda_device_context().stream()>>>( \
        dx_data, mul_grad.data<T>(), out->data<T>(), N, dim, dim); \
    break;

static inline int SizeOutAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = axis + 1; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
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

template <typename T, typename AccT, int Log2Elements>
__global__ void WarpSoftmaxForward(T* dst, const T* src, const int batch_size,
                                   const int stride, const int element_count) {
  constexpr int next_power_of_two = 1 << Log2Elements;
  constexpr int warp_size_softmax =
      (next_power_of_two < 32) ? next_power_of_two : 32;
  constexpr int WARP_ITERATIONS = next_power_of_two / warp_size_softmax;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH) {
    local_batches = WARP_BATCH;
  }

  int local_idx = threadIdx.x;

  src += first_batch * stride + local_idx;
  dst += first_batch * stride + local_idx;

  // load data from global memory
  AccT elements[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * warp_size_softmax;
      if (element_index < batch_element_count) {
        elements[i][it] =
            static_cast<float>(src[i * element_count + it * warp_size_softmax]);
      } else {
        elements[i][it] = -std::numeric_limits<AccT>::infinity();
      }
    }
  }

  // compute max_value
  AccT max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      max_value[i] =
          (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
    }
  }
  warp_reduce_max<AccT, WARP_BATCH, warp_size_softmax>(max_value);

  AccT sum[WARP_BATCH]{0.0f};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      elements[i][it] = (std::exp((elements[i][it] - max_value[i])));
      sum[i] += elements[i][it];
    }
  }
  warp_reduce_sum<AccT, WARP_BATCH, warp_size_softmax>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * warp_size_softmax;
      if (element_index < element_count) {
        dst[i * element_count + it * warp_size_softmax] =
            elements[i][it] / sum[i];
      } else {
        break;
      }
    }
  }
}

template <typename T, typename AccT, int Log2Elements>
__global__ void softmax_warp_backward(T* gradInput, const T* grad,
                                      const T* output, int batch_size,
                                      int stride, int element_count) {
  constexpr int next_power_of_two = 1 << Log2Elements;
  constexpr int warp_size_softmax =
      (next_power_of_two < 32) ? next_power_of_two : 32;
  constexpr int WARP_ITERATIONS = next_power_of_two / warp_size_softmax;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH) {
    local_batches = WARP_BATCH;
  }

  int local_idx = threadIdx.x % warp_size_softmax;

  int thread_offset = first_batch * stride + local_idx;
  grad += thread_offset;
  output += thread_offset;
  gradInput += thread_offset;

  // load data from global memory
  AccT grad_reg[WARP_BATCH][WARP_ITERATIONS];
  AccT output_reg[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * warp_size_softmax;
      if (element_index < batch_element_count) {
        grad_reg[i][it] =
            static_cast<AccT>(grad[i * element_count + it * warp_size_softmax]);
        output_reg[i][it] = static_cast<AccT>(
            output[i * element_count + it * warp_size_softmax]);
      } else {
        grad_reg[i][it] = AccT(0);
        output_reg[i][it] = AccT(0);
      }
    }
  }

  AccT sum[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    sum[i] = grad_reg[i][0];
#pragma unroll
    for (int it = 1; it < WARP_ITERATIONS; ++it) {
      sum[i] += grad_reg[i][it];
    }
  }
  warp_reduce_sum<AccT, WARP_BATCH, warp_size_softmax>(sum);

// store result
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * warp_size_softmax;
      if (element_index < element_count) {
        // compute gradients
        gradInput[i * element_count + it * warp_size_softmax] =
            (grad_reg[i][it] - output_reg[i][it] * sum[i]);
      }
    }
  }
}

template <typename T>
__global__ void MultiplyCUDAKernel(T* C, const T* A, const T* B, int N) {
  CUDA_KERNEL_LOOP(i, N) {
    C[i] = static_cast<T>(static_cast<float>(A[i]) * static_cast<float>(B[i]));
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

    constexpr int max_dim = 320;
    bool optimize = false;
    constexpr int warps_per_block = 4;
    if (D == 1 && dim <= max_dim && sizeof(T) <= 4) {
      if (dim == 128 && N % warps_per_block == 0) {
        optimize = true;
        // a warp for a batch, 4 elements for a thread, only support the softmax
        // dim size = 128 currently
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
      } else if (dim < max_dim) {
        optimize = true;
        int log2_elements = static_cast<int>(log2_ceil(dim));
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
          default:
            break;
        }
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
    bool optimize = false;
    if (D == 1 && warp_softmax_available) {
      if (dim == 128 && N % warps_per_block == 0) {
        optimize = true;
        if (std::is_same<T, float>::value) {
          VecSoftmaxBackward<float, 4, warps_per_block><<<
              N / warps_per_block, warps_per_block * WARP_SIZE, 0,
              ctx.cuda_device_context().stream()>>>(dx->data<float>(),
                                                    dout->data<float>(),
                                                    out->data<float>(), N, dim);
        } else if (std::is_same<T, platform::float16>::value) {
          VecSoftmaxBackward<platform::float16, 4, warps_per_block><<<
              N / warps_per_block, warps_per_block * WARP_SIZE, 0,
              ctx.cuda_device_context().stream()>>>(
              dx->data<platform::float16>(), dout->data<platform::float16>(),
              out->data<platform::float16>(), N, dim);
        } else {
          PADDLE_ENFORCE_EQ(
              warp_softmax_available, true,
              platform::errors::Unimplemented(
                  "Warp softmax backward is only available for fp32 and fp16"));
        }
      } else if (dim < 40 && dim % 32 != 0) {
        optimize = true;
        Tensor mul_grad;
        int numel = N * dim;
        mul_grad.mutable_data<T>({numel}, ctx.GetPlace());

        auto stream = ctx.cuda_device_context().stream();
        auto& dev_ctx =
            ctx.template device_context<platform::CUDADeviceContext>();
        auto config = GetGpuLaunchConfig1D(dev_ctx, numel);

        MultiplyCUDAKernel<T><<<config.block_per_grid.x,
                                config.thread_per_block.x, 0, stream>>>(
            mul_grad.data<T>(), dout->data<T>(), out->data<T>(), numel);

        int log2_elements = log2_ceil(dim);
        const int next_power_of_two = 1 << log2_elements;

        int warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (N + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);

        switch (log2_elements) {
          LAUNCH_SOFTMAX_WARP_BACKWARD(0);  // 1
          LAUNCH_SOFTMAX_WARP_BACKWARD(1);  // 2
          LAUNCH_SOFTMAX_WARP_BACKWARD(2);  // 4
          LAUNCH_SOFTMAX_WARP_BACKWARD(3);  // 8
          LAUNCH_SOFTMAX_WARP_BACKWARD(4);  // 16
          LAUNCH_SOFTMAX_WARP_BACKWARD(5);  // 32
          LAUNCH_SOFTMAX_WARP_BACKWARD(6);  // 64
          LAUNCH_SOFTMAX_WARP_BACKWARD(7);  // 128
          LAUNCH_SOFTMAX_WARP_BACKWARD(8);  // 256
          LAUNCH_SOFTMAX_WARP_BACKWARD(9);  // 512
          default:
            break;
        }
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
