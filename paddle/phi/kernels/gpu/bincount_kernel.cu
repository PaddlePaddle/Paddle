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
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

inline int GET_BLOCKS(const int N) {
  return (N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
}

template <typename T, typename InputT, typename OutT>
__global__ void KernelBincount(const InputT* input,
                               const int total_elements,
                               const bool has_weights,
                               const T* weights,
                               OutT* output) {
  if (!has_weights) {
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
      phi::CudaAtomicAdd(&output[input[i]], 1L);
    }
  } else {
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
      phi::CudaAtomicAdd(&output[input[i]], static_cast<OutT>(weights[i]));
    }
  }
}

template <typename Context, typename T, typename InputT>
void BincountCUDAInner(const Context& dev_ctx,
                       const DenseTensor& x,
                       const paddle::optional<DenseTensor>& weights,
                       int minlength,
                       DenseTensor* out) {
  const DenseTensor* input = &x;
  DenseTensor* output = out;
  const InputT* input_data = input->data<InputT>();

  const int input_numel = input->numel();

  if (input_data == nullptr) {
    phi::DDim out_dim{0};
    output->Resize(out_dim);
    dev_ctx.template Alloc<T>(output);
    return;
  }
  auto input_x = EigenVector<InputT>::Flatten(*input);
  DenseTensor input_min_t, input_max_t;
  input_max_t.Resize({1});
  auto* input_max_data = dev_ctx.template Alloc<InputT>(&input_max_t);
  input_min_t.Resize({1});
  auto* input_min_data = dev_ctx.template Alloc<InputT>(&input_min_t);

  auto input_max_scala = EigenScalar<InputT>::From(input_max_t);
  auto input_min_scala = EigenScalar<InputT>::From(input_min_t);

  auto* place = dev_ctx.eigen_device();
  input_max_scala.device(*place) = input_x.maximum();
  input_min_scala.device(*place) = input_x.minimum();

  DenseTensor input_min_cpu, input_max_cpu;
  paddle::framework::TensorCopySync(
      input_max_t, phi::CPUPlace(), &input_max_cpu);
  paddle::framework::TensorCopySync(
      input_min_t, phi::CPUPlace(), &input_min_cpu);

  InputT input_min = input_min_cpu.data<InputT>()[0];

  PADDLE_ENFORCE_GE(
      input_min,
      static_cast<InputT>(0),
      phi::errors::InvalidArgument(
          "The elements in input tensor must be non-negative ints"));

  int64_t output_size =
      static_cast<int64_t>(input_max_cpu.data<InputT>()[0]) + 1L;

  output_size = std::max(output_size, static_cast<int64_t>(minlength));
  phi::DDim out_dim{output_size};
  output->Resize(out_dim);

  bool has_weights = weights.is_initialized();

  const T* weights_data = has_weights ? weights->data<T>() : nullptr;
  auto stream = dev_ctx.stream();

  if (!has_weights) {
    int64_t* output_data = dev_ctx.template Alloc<int64_t>(output);
    phi::funcs::SetConstant<Context, int64_t>()(dev_ctx, output, 0L);

    KernelBincount<T, InputT, int64_t>
        <<<GET_BLOCKS(input_numel), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
            input_data, input_numel, has_weights, weights_data, output_data);
  } else {
    const auto& weights_type =
        paddle::framework::TransToProtoVarType(weights->dtype());

    if (weights->dtype() == DataType::FLOAT32) {
      float* output_data = dev_ctx.template Alloc<float>(output);
      phi::funcs::SetConstant<Context, float>()(
          dev_ctx, output, static_cast<float>(0));

      KernelBincount<T, InputT, float>
          <<<GET_BLOCKS(input_numel), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
              input_data, input_numel, has_weights, weights_data, output_data);
    } else {
      double* output_data = dev_ctx.template Alloc<double>(output);
      phi::funcs::SetConstant<Context, double>()(
          dev_ctx, output, static_cast<double>(0));
      KernelBincount<T, InputT, double>
          <<<GET_BLOCKS(input_numel), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
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
  int int_minlength = minlength.to<int>();
  PADDLE_ENFORCE_GE(int_minlength,
                    0,
                    phi::errors::InvalidArgument(
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
                   int64_t) {}
