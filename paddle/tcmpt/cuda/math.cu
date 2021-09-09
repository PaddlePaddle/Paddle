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

#include "paddle/tcmpt/cuda/math.h"

// #include "paddle/tcmpt/eigen/scale.h"
// #include "paddle/tcmpt/eigen/sign.h"

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
  module::Sign<CUDAContext, T>(dev_ctx, x, out);
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
  module::Scale<CUDAContext, T>(dev_ctx, x, scale, bias, bias_after_scale, out);
}

template <typename T>
void ScaleSelectedRows(const CUDAContext& dev_ctx,
                       const SelectedRowsTensor& x,
                       float scale,
                       float bias,
                       bool bias_after_scale,
                       SelectedRowsTensor* out) {
  out->set_rows(x.rows());
  out->set_height(x.height());
  Scale<T>(
      dev_ctx, x.value(), scale, bias, bias_after_scale, out->mutable_value());
}

template <typename T>
void ScaleDynamicAttr(const CUDAContext& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& scale,
                      float bias,
                      bool bias_after_scale,
                      DenseTensor* out) {
  module::Scale<CUDAContext, T>(
      dev_ctx, x, *scale.data<float>(), bias, bias_after_scale, out);
}

template <typename T>
void ScaleSelectedRowsDynamicAttr(const CUDAContext& dev_ctx,
                                  const SelectedRowsTensor& x,
                                  const DenseTensor& scale,
                                  float bias,
                                  bool bias_after_scale,
                                  SelectedRowsTensor* out) {
  out->set_rows(x.rows());
  out->set_height(x.height());
  Scale<T>(dev_ctx,
           x.value(),
           *scale.data<float>(),
           bias,
           bias_after_scale,
           out->mutable_value());
}

}  // namespace pt

using float16 = paddle::platform::float16;
PT_REGISTER_KERNEL("sign", CUDA, NCHW, pt::Sign, float, double, float16) {}
PT_REGISTER_KERNEL("mean", CUDA, NCHW, pt::Mean, float, double, float16) {}
PT_REGISTER_KERNEL("scale",
                   CUDA,
                   NCHW,
                   pt::Scale,
                   float,
                   double,
                   float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
PT_REGISTER_KERNEL("scale.selectedrows",
                   CUDA,
                   NCHW,
                   pt::ScaleSelectedRows,
                   float,
                   double,
                   float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
PT_REGISTER_KERNEL("scale.dynamic_attr",
                   CUDA,
                   NCHW,
                   pt::ScaleDynamicAttr,
                   float,
                   double,
                   float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1)
      .SetBackend(pt::Backend::kCPU)
      .SetDataType(pt::DataType::kFLOAT32);
}
PT_REGISTER_KERNEL("scale.selectedrows.dynamic_attr",
                   CUDA,
                   NCHW,
                   pt::ScaleSelectedRowsDynamicAttr,
                   float,
                   double,
                   float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1)
      .SetBackend(pt::Backend::kCPU)
      .SetDataType(pt::DataType::kFLOAT32);
}
