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

#include "paddle/tcmpt/kernels/cuda/math.h"

#include "paddle/tcmpt/kernels/common/eigen/mean.h"
#include "paddle/tcmpt/kernels/common/eigen/scale.h"
#include "paddle/tcmpt/kernels/common/eigen/sign.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/platform/float16.h"
#include "paddle/tcmpt/core/convert_utils.h"
#include "paddle/tcmpt/core/kernel_registry.h"

namespace pt {

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
  VLOG(1) << "chenweihang: call new pt mean kernel.";
  // eigen::Mean<CUDAContext, T>(dev_ctx, x, out);
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

  pt::DenseTensor tmp(
      TensorMeta(paddle::framework::make_ddim(
                     {static_cast<int64_t>(temp_storage_bytes)}),
                 pt::TransToPtBackend(dev_ctx.GetPlace()),
                 x.type(),
                 x.layout()),
      TensorStatus());
  auto* temp_storage = tmp.mutable_data<uint8_t>();
  err = cub::DeviceReduce::Sum(
      temp_storage, temp_storage_bytes, trans_x, out_data, size_prob, stream);
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
  if (paddle::platform::is_gpu_place(scale.place())) {
    throw std::runtime_error("scale host place error.");
  }
  eigen::Scale<CUDAContext, T>(dev_ctx,
                               x,
                               static_cast<float>(*scale.data<T>()),
                               bias,
                               bias_after_scale,
                               out);
}

}  // namespace pt

// TODO(chenweihang): replace by better impl
PT_REGISTER_MODULE(MathCUDA);

using float16 = paddle::platform::float16;
PT_REGISTER_KERNEL("sign", CUDA, Any, pt::Sign, float, double, float16) {}
PT_REGISTER_KERNEL("mean", CUDA, Any, pt::Mean, float, double, float16) {}
PT_REGISTER_KERNEL("scale",
                   CUDA,
                   Any,
                   pt::Scale,
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
                   Any,
                   pt::ScaleHost,
                   float,
                   double,
                   float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetBackend(pt::Backend::kCPU);
}
