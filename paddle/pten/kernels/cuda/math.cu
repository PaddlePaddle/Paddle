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

#include "paddle/pten/kernels/cuda/math.h"

#include "paddle/pten/kernels/functions/cuda/elementwise/elementwise.h"
#include "paddle/pten/kernels/functions/eigen/mean.h"
#include "paddle/pten/kernels/functions/eigen/scale.h"
#include "paddle/pten/kernels/functions/eigen/sign.h"
#include "paddle/pten/kernels/functions/general/elementwise_functor.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {

/**
 * Util Functors
 */

template <typename T>
struct DivideFunctor {
  HOSTDEVICE explicit inline DivideFunctor(int n)
      : n_inv(static_cast<T>(1.0 / n)) {}

  HOSTDEVICE inline T operator()(const T& x) const { return x * n_inv; }

 private:
  T n_inv;
};

/**
 * Kernels
 */

template <typename T>
void Sign(const CUDAContext& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  eigen::Sign<CUDAContext, T>(dev_ctx, x, out);
}

template <typename T>
void Mean(const CUDAContext& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  auto size_prob = x.numel();
  const T* x_data = x.data<T>();
  T* out_data = out->mutable_data<T>();
  auto stream = dev_ctx.stream();

  DivideFunctor<T> transformer(size_prob);
  cub::TransformInputIterator<T, DivideFunctor<T>, const T*> trans_x(
      x_data, transformer);
  size_t temp_storage_bytes = 0;

  auto err = cub::DeviceReduce::Sum(
      nullptr, temp_storage_bytes, trans_x, out_data, size_prob, stream);
  PADDLE_ENFORCE_CUDA_SUCCESS(err);

  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      dev_ctx.GetPlace());
  pten::DenseTensor tmp(
      alloc,
      DenseTensorMeta(x.dtype(),
                      paddle::framework::make_ddim(
                          {static_cast<int64_t>(temp_storage_bytes)}),
                      x.layout()));
  void* temp_storage = tmp.mutable_data<T>();
  err = cub::DeviceReduce::Sum(static_cast<uint8_t*>(temp_storage),
                               temp_storage_bytes,
                               trans_x,
                               out_data,
                               size_prob,
                               stream);
  PADDLE_ENFORCE_CUDA_SUCCESS(err);
}

template <typename T>
void Scale(const CUDAContext& dev_ctx,
           const DenseTensor& x,
           float scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out) {
  eigen::Scale<CUDAContext, T>(dev_ctx, x, scale, bias, bias_after_scale, out);
}

template <typename T>
void ScaleHost(const CUDAContext& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& scale,
               float bias,
               bool bias_after_scale,
               DenseTensor* out) {
  PADDLE_ENFORCE_EQ(paddle::platform::is_gpu_place(scale.place()),
                    false,
                    paddle::platform::errors::InvalidArgument(
                        "Scale argument isn't a host tensor."));
  eigen::Scale<CUDAContext, T>(dev_ctx,
                               x,
                               static_cast<float>(*scale.data<T>()),
                               bias,
                               bias_after_scale,
                               out);
}

template <typename T>
void ElementwiseAdd(const CUDAContext& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out) {
  std::vector<const DenseTensor*> inputs;
  std::vector<DenseTensor*> outputs;
  inputs.emplace_back(&x);
  inputs.emplace_back(&y);
  outputs.emplace_back(out);
  LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
      dev_ctx, inputs, &outputs, axis, general::AddFunctor<T>());
}

template <typename T>
void ElementwiseSub(const CUDAContext& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out) {
  std::vector<const DenseTensor*> inputs;
  std::vector<DenseTensor*> outputs;
  inputs.emplace_back(&x);
  inputs.emplace_back(&y);
  outputs.emplace_back(out);
  LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
      dev_ctx, inputs, &outputs, axis, general::SubFunctor<T>());
}

}  // namespace pten

// TODO(chenweihang): replace by better impl
PT_REGISTER_MODULE(MathCUDA);

using float16 = paddle::platform::float16;
using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_KERNEL("sign", CUDA, ANY, pten::Sign, float, double, float16) {}
PT_REGISTER_KERNEL("mean", CUDA, ANY, pten::Mean, float, double, float16) {}
PT_REGISTER_KERNEL("scale",
                   CUDA,
                   ANY,
                   pten::Scale,
                   float,
                   double,
                   float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
PT_REGISTER_KERNEL("scale.host",
                   CUDA,
                   ANY,
                   pten::ScaleHost,
                   float,
                   double,
                   float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetBackend(pten::Backend::CPU);
}
PT_REGISTER_KERNEL("elementwise_add",
                   CUDA,
                   ANY,
                   pten::ElementwiseAdd,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL("elementwise_sub",
                   CUDA,
                   ANY,
                   pten::ElementwiseSub,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   complex64,
                   complex128) {}
