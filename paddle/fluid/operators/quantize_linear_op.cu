/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/quantize_linear_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void KeDequantize(const T* in, const T* scale, T max_range,
                             int64_t num, T* out) {
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int64_t i = idx; i < num; i += blockDim.x * gridDim.x) {
    out[i] = in[i] * scale[0] / max_range;
  }
}

template <typename T>
struct DequantizeFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& dev_ctx,
                  const framework::Tensor* in, const framework::Tensor* scale,
                  T max_range, framework::Tensor* out) {
    const T* in_data = in->data<T>();
    const T* scale_factor = scale->data<T>();
    T* out_data = out->mutable_data<T>(dev_ctx.GetPlace());

    int64_t num = in->numel();
    int64_t block_size = std::min(
        num, static_cast<int64_t>(dev_ctx.GetMaxThreadsPerBlock() / 4));
    int64_t max_threads =
        dev_ctx.GetMaxPhysicalThreadCount();  // SM * block_per_SM
    const int64_t max_blocks =
        std::max(((max_threads - 1) / block_size + 1), static_cast<int64_t>(1));
    const int64_t grid_size =
        std::min(max_blocks, (num + block_size - 1) / block_size);
    KeDequantize<T><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
        in_data, scale_factor, max_range, num, out_data);
  }
};

template <typename T>
__global__ void DequantizeOneScaleQuantAxisN(const T* in, const T* scale,
                                             const T max_range,
                                             const int64_t num,
                                             const int n_scales,
                                             const int quant_stride, T* out) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int64_t i = idx; i < num; i += blockDim.x * gridDim.x) {
    T s = scale[(i / quant_stride) % n_scales];
    out[i] = in[i] * s / max_range;
  }
}

template <typename T>
struct ChannelDequantizeFunctorV2<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& dev_ctx,
                  const framework::Tensor* in, const framework::Tensor* scale,
                  T max_range, const int quant_axis, framework::Tensor* out) {
    auto in_dims = in->dims();
    const T* in_data = in->data<T>();
    T* out_data = out->mutable_data<T>(dev_ctx.GetPlace());
    int64_t num = in->numel();
    const T* scale_factor = scale->data<T>();
    int64_t block_size = std::min(
        num, static_cast<int64_t>(dev_ctx.GetMaxThreadsPerBlock() / 4));
    int64_t max_threads =
        dev_ctx.GetMaxPhysicalThreadCount();  // SM * block_per_SM
    const int64_t max_blocks =
        std::max(((max_threads - 1) / block_size + 1), static_cast<int64_t>(1));
    const int64_t grid_size =
        std::min(max_blocks, (num + block_size - 1) / block_size);

    int quant_stride = 1;
    for (int i = quant_axis + 1; i < in_dims.size(); i++) {
      quant_stride *= in_dims[i];
    }

    DequantizeOneScaleQuantAxisN<
        T><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
        in_data, scale_factor, max_range, num, in_dims[quant_axis],
        quant_stride, out_data);
  }
};

template struct DequantizeFunctor<platform::CUDADeviceContext, float>;
template struct DequantizeFunctor<platform::CUDADeviceContext, double>;
template struct ChannelDequantizeFunctorV2<platform::CUDADeviceContext, float>;
template struct ChannelDequantizeFunctorV2<platform::CUDADeviceContext, double>;

template <typename T>
__global__ void FindAbsMaxKernel(const T* in, const int n, T* out) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  extern __shared__ char* shared_max_data_tmp[];
  auto shared_max_data = reinterpret_cast<T*>(shared_max_data_tmp);
  if (gridDim.x > 1) {
    T local_max_data = T(0);
    for (int i = bid; i < n; i += blockDim.x * gridDim.x) {
      T tmp = abs(in[i]);
      if (tmp > local_max_data) {
        local_max_data = tmp;
      }
    }
    shared_max_data[tid] = local_max_data;
  } else {
    if (bid < n) {
      shared_max_data[tid] = abs(in[bid]);
    } else {
      shared_max_data[tid] = T(0);
    }
  }
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i && (shared_max_data[tid] < shared_max_data[tid + i])) {
      shared_max_data[tid] = shared_max_data[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = shared_max_data[0];
  }
}

template <typename T>
struct FindAbsMaxFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx, const T* in,
                  const int num, T* out) {
    int block = 1024;
    int grid = (block - 1 + num) / block;
    grid = (grid > block) ? block : grid;

    framework::Tensor max;
    T* max_data = max.mutable_data<T>(phi::make_ddim({grid}), ctx.GetPlace());
    FindAbsMaxKernel<T><<<grid, block, 1024 * sizeof(T), ctx.stream()>>>(
        in, num, max_data);
    FindAbsMaxKernel<T><<<1, block, 1024 * sizeof(T), ctx.stream()>>>(
        max_data, grid, out);
  }
};

template struct FindAbsMaxFunctor<platform::CUDADeviceContext, float>;
template struct FindAbsMaxFunctor<platform::CUDADeviceContext,
                                  paddle::platform::float16>;

template <typename T>
__global__ void FindChannelAbsMaxKernelQuantAxis0(const T* in, const int n,
                                                  const int c, T* out) {
  int tid = threadIdx.x;
  int channel_size = n / c;
  const T* in_c = in + blockIdx.x * channel_size;
  extern __shared__ T shared_max_data[];
  shared_max_data[tid] = T(0);
  for (int i = tid; i < channel_size; i += blockDim.x) {
    T tmp = fabs(in_c[i]);
    if (tmp > shared_max_data[tid]) {
      shared_max_data[tid] = tmp;
    }
  }
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i && (shared_max_data[tid] < shared_max_data[tid + i])) {
      shared_max_data[tid] = shared_max_data[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = shared_max_data[0];
  }
}

template <typename T>
__global__ void FindChannelAbsMaxKernelQuantAxis1(const T* in, const int n,
                                                  const int cin, const int cout,
                                                  T* out) {
  extern __shared__ T shared_max_data[];
  int cout_wh_size = n / cin;
  int wh_size = n / (cin * cout);

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  const T* in_current = in + tid * cout_wh_size + bid * wh_size;
  T local_max_data = T(0);
  for (int i = 0; i < wh_size; i++) {
    T tmp = fabs(in_current[i]);
    if (tmp > local_max_data) {
      local_max_data = tmp;
    }
  }
  shared_max_data[tid] = local_max_data;
  __syncthreads();

  int len = blockDim.x;
  for (int i = (len + 1) / 2; i > 0; len = i, i = (i + 1) / 2) {
    if (tid < i && tid + i < len &&
        shared_max_data[tid] < shared_max_data[tid + i]) {
      shared_max_data[tid] = shared_max_data[tid + i];
    }
    if (i == 1) {
      i = 0;  // break the loop
    }
    __syncthreads();
  }
  if (tid == 0 && shared_max_data[0] > out[bid]) {
    out[bid] = shared_max_data[0];
  }
}

template <typename T>
struct FindChannelAbsMaxFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& in_tensor, const int quant_axis,
                  T* out_abs_max) {
    PADDLE_ENFORCE_EQ(
        quant_axis == 0 || quant_axis == 1, true,
        platform::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                          "the received is %d",
                                          quant_axis));
    const int num = in_tensor.numel();
    auto in_dims = in_tensor.dims();
    const T* in_data = in_tensor.data<T>();
    if (quant_axis == 0) {
      int cout = in_dims[0];
      int grid = cout;
      int block = 1024;
      FindChannelAbsMaxKernelQuantAxis0<
          T><<<grid, block, block * sizeof(T), ctx.stream()>>>(
          in_data, num, cout, out_abs_max);
    } else if (quant_axis == 1) {
      int cin = in_dims[0];
      int cout = in_dims[1];
      int grid = cout;
      int max_threads = 1024;

#ifdef PADDLE_WITH_HIP
      hipMemset(out_abs_max, 0, sizeof(T) * cout);
#else
      cudaMemset(out_abs_max, 0, sizeof(T) * cout);
#endif

      for (int i = 0; i < cin / max_threads; i++) {
        int block = max_threads;
        FindChannelAbsMaxKernelQuantAxis1<
            T><<<grid, block, block * sizeof(T), ctx.stream()>>>(
            in_data, num, cin, cout, out_abs_max);
        in_data += num / cin;
      }

      int block = cin % max_threads;
      if (block > 0) {
        FindChannelAbsMaxKernelQuantAxis1<
            T><<<grid, block, block * sizeof(T), ctx.stream()>>>(
            in_data, num, in_dims[0], in_dims[1], out_abs_max);
      }
    }
  }
};

template struct FindChannelAbsMaxFunctor<platform::CUDADeviceContext, float>;

template <typename T>
__global__ void ClipAndQuantKernel(const T* in, const T* scale,
                                   const int bin_cnt, const int n, T* out) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  T s = scale[0];
  T inv_s = inverse(s);
  for (int i = bid; i < n; i += blockDim.x * gridDim.x) {
    T x = in[i];
    T v = x > s ? s : x;
    v = v < -s ? -s : v;
    v = bin_cnt * inv_s * v;
    out[i] = round(v);
  }
}

template <typename T>
struct ClipAndFakeQuantFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& scale,
                  const int bin_cnt, framework::Tensor* out) {
    int num = in.numel();
    int block = 1024;
    int grid = (block - 1 + num) / block;

    const T* in_data = in.data<T>();
    const T* scale_data = scale.data<T>();
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    ClipAndQuantKernel<T><<<grid, block, 0, ctx.stream()>>>(
        in_data, scale_data, bin_cnt, num, out_data);
  }
};

template struct ClipAndFakeQuantFunctor<platform::CUDADeviceContext, float>;

// ChannelClipAndQuantKernel for quant_axis is 0
template <typename T>
__global__ void ChannelClipAndQuantKernelQuantAxis0(const T* in, const T* scale,
                                                    const int bin_cnt,
                                                    const int64_t n,
                                                    const int c, T* out) {
  int tid = threadIdx.x;

  int64_t channel_size = n / c;
  const T* in_c = in + blockIdx.x * channel_size;
  T* out_c = out + blockIdx.x * channel_size;

  T s = scale[blockIdx.x];
  T inv_s = inverse(s);

  for (int64_t i = tid; i < channel_size; i += blockDim.x) {
    T x = in_c[i];
    T v = x > s ? s : x;
    v = v < -s ? -s : v;
    v = bin_cnt * inv_s * v;
    out_c[i] = round(v);
  }
}

// ChannelClipAndQuantKernel for quant_axis is N
template <typename T>
__global__ void ChannelClipAndQuantKernelQuantAxisN(
    const T* in, const T* scale, const int bin_cnt, const int64_t n,
    const int nScale, const int quant_stride, T* out) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int64_t i = idx; i < n; i += blockDim.x * gridDim.x) {
    T s = scale[(i / quant_stride) % nScale];
    T inv_s = inverse(s);
    T x = in[i];
    T v = x > s ? s : x;
    v = v < -s ? -s : v;
    v = bin_cnt * inv_s * v;
    out[i] = round(v);
  }
}

template <typename T>
struct ChannelClipAndFakeQuantFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& scale,
                  const int bin_cnt, const int quant_axis,
                  framework::Tensor* out) {
    PADDLE_ENFORCE_EQ(
        quant_axis == 0 || quant_axis == 1, true,
        platform::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                          "the received is %d",
                                          quant_axis));

    int64_t num = in.numel();
    auto in_dims = in.dims();
    const T* in_data = in.data<T>();
    const T* scale_data = scale.data<T>();
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    if (quant_axis == 0) {
      int grid = in_dims[0];
      int block = 1024;
      ChannelClipAndQuantKernelQuantAxis0<T><<<grid, block, 0, ctx.stream()>>>(
          in_data, scale_data, bin_cnt, num, in_dims[0], out_data);
    } else {
      int quant_stride = 1;
      for (int i = quant_axis + 1; i < in_dims.size(); i++) {
        quant_stride *= in_dims[i];
      }
      int64_t block_size =
          std::min(num, static_cast<int64_t>(ctx.GetMaxThreadsPerBlock() / 4));
      int64_t max_threads =
          ctx.GetMaxPhysicalThreadCount();  // SM * block_per_SM
      const int64_t max_blocks = std::max(((max_threads - 1) / block_size + 1),
                                          static_cast<int64_t>(1));

      const int64_t grid_size =
          std::min(max_blocks, (num + block_size - 1) / block_size);

      ChannelClipAndQuantKernelQuantAxisN<T><<<grid_size, block_size>>>(
          in_data, scale_data, bin_cnt, num, in_dims[quant_axis], quant_stride,
          out_data);
    }
  }
};

template struct ChannelClipAndFakeQuantFunctor<platform::CUDADeviceContext,
                                               float>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(dequantize_linear,
                        ops::DeQuantizeLinearKernel<CUDA, float, float>,
                        ops::DeQuantizeLinearKernel<CUDA, int8_t, float>,
                        ops::DeQuantizeLinearKernel<CUDA, double, double>);

REGISTER_OP_CUDA_KERNEL(quantize_linear,
                        ops::QuantizeLinearKernel<CUDA, float>);
