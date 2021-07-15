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

#include "paddle/pten/cuda/math.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/pten/core/convert_utils.h"

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
void MeanCUDA(const CUDADeviceContext& dev_ctx,
              const BaseTensor& x,
              BaseTensor* out) {
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

  // TODO(chenweihang): maybe too complicated
  pt::TensorMeta meta(
      paddle::framework::make_ddim({static_cast<int64_t>(temp_storage_bytes)}),
      pt::TransToPtenBackend(dev_ctx.GetPlace()),
      x.type(),
      x.layout(),
      0);
  pt::BaseTensor tmp(std::move(meta));
  auto* temp_storage = tmp.mutable_data<uint8_t>();
  err = cub::DeviceReduce::Sum(
      temp_storage, temp_storage_bytes, trans_x, out_data, size_prob, stream);
  PADDLE_ENFORCE_CUDA_SUCCESS(err);
}

template void MeanCUDA<float>(const CUDADeviceContext& dev_ctx,
                              const BaseTensor& x,
                              BaseTensor* out);
template void MeanCUDA<double>(const CUDADeviceContext& dev_ctx,
                               const BaseTensor& x,
                               BaseTensor* out);
template void MeanCUDA<paddle::platform::float16>(
    const CUDADeviceContext& dev_ctx, const BaseTensor& x, BaseTensor* out);

}  // namespace pt
