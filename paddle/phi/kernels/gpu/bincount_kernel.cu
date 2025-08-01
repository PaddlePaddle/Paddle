// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/bincount_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"
namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

inline int64_t GET_BLOCKS(const int64_t N) {
  return (N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
}

template <typename T>
__global__ void KernelReduceMinMax(const T* input,
                                   int64_t numel,
                                   T* min_out,
                                   T* max_out) {
  __shared__ T smin[PADDLE_CUDA_NUM_THREADS];
  __shared__ T smax[PADDLE_CUDA_NUM_THREADS];
  int tid = threadIdx.x;
  int64_t global_thread_id =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  T local_min = std::numeric_limits<T>::max();
  T local_max = std::numeric_limits<T>::lowest();

  for (int64_t i = global_thread_id; i < numel; i += stride) {
    T val = input[i];
    local_min = min(local_min, val);
    local_max = max(local_max, val);
  }

  smin[tid] = local_min;
  smax[tid] = local_max;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      smin[tid] = min(smin[tid], smin[tid + offset]);
      smax[tid] = max(smax[tid], smax[tid + offset]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    phi::CudaAtomicMin(min_out, smin[0]);
    phi::CudaAtomicMax(max_out, smax[0]);
  }
}

template <typename T, typename InputT, typename OutT>
__global__ void KernelBincount(const InputT* input,
                               const int64_t total_elements,
                               const bool has_weights,
                               const T* weights,
                               OutT* output) {
  int64_t global_tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = global_tid; i < total_elements; i += stride) {
    InputT index = input[i];
    if (!has_weights) {
      phi::CudaAtomicAdd(&output[index], 1L);
    } else {
      phi::CudaAtomicAdd(&output[index], static_cast<OutT>(weights[i]));
    }
  }
}

template <typename Context, typename T, typename InputT>
void BincountCUDAInner(const Context& dev_ctx,
                       const DenseTensor& x,
                       const paddle::optional<DenseTensor>& weights,
                       int64_t minlength,
                       DenseTensor* out) {
  const DenseTensor* input = &x;
  DenseTensor* output = out;
  const InputT* input_data = input->data<InputT>();

  int64_t input_numel = static_cast<int64_t>(input->numel());

  if (input_data == nullptr) {
    phi::DDim out_dim{minlength};
    output->Resize(out_dim);
    phi::Full<int64_t, Context>(
        dev_ctx, phi::IntArray(common::vectorize(output->dims())), 0, output);
    return;
  }

  DenseTensor input_min_max_cpu;
  input_min_max_cpu.Resize({2});
  auto* input_min_max_cpu_data =
      dev_ctx.template HostAlloc<InputT>(&input_min_max_cpu);
  input_min_max_cpu.data<InputT>()[0] = std::numeric_limits<InputT>::max();
  input_min_max_cpu.data<InputT>()[1] = std::numeric_limits<InputT>::lowest();

  DenseTensor input_min_max_t;
  input_min_max_t.Resize({2});
  auto* input_min_max_data = dev_ctx.template Alloc<InputT>(&input_min_max_t);

  phi::Copy(
      dev_ctx, input_min_max_cpu, dev_ctx.GetPlace(), true, &input_min_max_t);

  int64_t max_grid_x = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int64_t num_blocks = std::min(GET_BLOCKS(input_numel), max_grid_x);
  KernelReduceMinMax<InputT>
      <<<num_blocks, PADDLE_CUDA_NUM_THREADS, 0, dev_ctx.stream()>>>(
          input_data, input_numel, input_min_max_data, input_min_max_data + 1);

  phi::Copy(
      dev_ctx, input_min_max_t, phi::CPUPlace(), true, &input_min_max_cpu);

  InputT input_min = input_min_max_cpu.data<InputT>()[0];

  PADDLE_ENFORCE_GE(
      input_min,
      static_cast<InputT>(0),
      common::errors::InvalidArgument(
          "The elements in input tensor must be non-negative ints"));

  int64_t output_size =
      static_cast<int64_t>(input_min_max_cpu.data<InputT>()[1]) + 1L;

  output_size = std::max(output_size, minlength);
  phi::DDim out_dim{output_size};
  output->Resize(out_dim);

  bool has_weights = weights.is_initialized();

  const T* weights_data = has_weights ? weights->data<T>() : nullptr;
  auto stream = dev_ctx.stream();

  if (!has_weights) {
    int64_t* output_data = dev_ctx.template Alloc<int64_t>(output);
    phi::funcs::SetConstant<Context, int64_t>()(
        dev_ctx, output, static_cast<int64_t>(0));

    KernelBincount<T, InputT, int64_t>
        <<<num_blocks, PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
            input_data, input_numel, has_weights, weights_data, output_data);
  } else {
    if (weights->dtype() == DataType::FLOAT32) {
      float* output_data = dev_ctx.template Alloc<float>(output);
      phi::funcs::SetConstant<Context, float>()(
          dev_ctx, output, static_cast<float>(0));

      KernelBincount<T, InputT, float>
          <<<num_blocks, PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
              input_data, input_numel, has_weights, weights_data, output_data);
    } else {
      double* output_data = dev_ctx.template Alloc<double>(output);
      phi::funcs::SetConstant<Context, double>()(
          dev_ctx, output, static_cast<double>(0));
      KernelBincount<T, InputT, double>
          <<<num_blocks, PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
              input_data, input_numel, has_weights, weights_data, output_data);
    }
  }
}

template <typename T, typename Context>
void BincountKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const paddle::optional<DenseTensor>& weights,
                    const Scalar& minlength,
                    DenseTensor* out) {
  int64_t int_minlength = minlength.to<int64_t>();
  PADDLE_ENFORCE_GE(int_minlength,
                    0,
                    common::errors::InvalidArgument(
                        "The minlength should be greater than or equal to 0."
                        "But received minlength is %d",
                        int_minlength));

  if (x.dtype() == DataType::INT32) {
    BincountCUDAInner<Context, T, int>(dev_ctx, x, weights, int_minlength, out);
  } else if (x.dtype() == DataType::INT64) {
    BincountCUDAInner<Context, T, int64_t>(
        dev_ctx, x, weights, int_minlength, out);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(bincount,
                   GPU,
                   ALL_LAYOUT,
                   phi::BincountKernel,
                   float,
                   double,
                   int,
                   int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
