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

#include "paddle/phi/kernels/arange_kernel.h"

#include "paddle/common/errors.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/range_function.h"

namespace phi {

template <typename T, typename OUT_TYPE>
__global__ void Range(T start, T step, int64_t size, OUT_TYPE* out) {
  CUDA_KERNEL_LOOP_TYPE(index, size, int64_t) {
    out[index] = static_cast<OUT_TYPE>(start + step * index);
  }
}

template <typename T, typename Context>
void ArangeTensorKernel(const Context& dev_ctx,
                        const DenseTensor& start,
                        const DenseTensor& end,
                        const DenseTensor& step,
                        DenseTensor* out) {
  bool any_float = phi::IsFloatingType(start.dtype()) ||
                   phi::IsFloatingType(end.dtype()) ||
                   phi::IsFloatingType(step.dtype());
  int64_t size = 0;
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  Scalar start_scalar(start);
  Scalar end_scalar(end);
  Scalar step_scalar(step);
  if (any_float) {
    double sv = start_scalar.to<double>();
    double ev = end_scalar.to<double>();
    double stv = step_scalar.to<double>();
    funcs::GetSize<double>(sv, ev, stv, &size);
  } else {
    int64_t sv = start_scalar.to<int64_t>();
    int64_t ev = end_scalar.to<int64_t>();
    int64_t stv = step_scalar.to<int64_t>();
    funcs::GetSize<int64_t>(sv, ev, stv, &size);
  }
  MPType start_value = start_scalar.to<MPType>();
  MPType step_value = step_scalar.to<MPType>();

  out->Resize({size});
  T* out_data = dev_ctx.template Alloc<T>(out);

  auto stream = dev_ctx.stream();
  int64_t block = std::min(size, static_cast<int64_t>(256));
  if (block == 0) {
    return;
  }
  int64_t grid = (size + block - 1) / block;
  Range<MPType, T>
      <<<grid, block, 0, stream>>>(start_value, step_value, size, out_data);
}

template <typename T, typename Context>
void ArangeNullaryKernel(const Context& dev_ctx,
                         const T start_value,
                         const T end_value,
                         const T step_value,
                         DenseTensor* out) {
  using MT = typename MPTypeTrait<T>::Type;
  MT start_value_mpt = static_cast<MT>(start_value);
  MT end_value_mpt = static_cast<MT>(end_value);
  MT step_value_mpt = static_cast<MT>(step_value);
  int64_t size = 0;
  funcs::GetSize(start_value_mpt, end_value_mpt, step_value_mpt, &size);
  out->Resize({size});
  T* out_data = dev_ctx.template Alloc<T>(out);

  auto stream = dev_ctx.stream();
  int64_t block = std::min(size, static_cast<int64_t>(256));
  if (block == 0) {
    return;
  }
  int64_t grid = (size + block - 1) / block;
  Range<MT, T><<<grid, block, 0, stream>>>(
      start_value_mpt, step_value_mpt, size, out_data);
}

template <typename T, typename Context>
void ArangeKernel(const Context& dev_ctx,
                  const Scalar& start,
                  const Scalar& end,
                  const Scalar& step,
                  DenseTensor* out) {
  bool is_floating = phi::IsFloatingType(start.dtype()) ||
                     phi::IsFloatingType(end.dtype()) ||
                     phi::IsFloatingType(step.dtype());
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  int64_t size = 0;
  if (is_floating) {
    double sv = start.to<double>();
    double ev = end.to<double>();
    double stv = step.to<double>();
    funcs::GetSize<double>(sv, ev, stv, &size);
  } else {
    int64_t sv = start.to<int64_t>();
    int64_t ev = end.to<int64_t>();
    int64_t stv = step.to<int64_t>();
    funcs::GetSize<int64_t>(sv, ev, stv, &size);
  }
  MPType start_value = start.to<MPType>();
  MPType step_value = step.to<MPType>();
  out->Resize({size});
  T* out_data = dev_ctx.template Alloc<T>(out);

  auto stream = dev_ctx.stream();
  int64_t block = std::min(size, static_cast<int64_t>(256));
  if (block == 0) {
    return;
  }
  int64_t grid = (size + block - 1) / block;
  Range<MPType, T>
      <<<grid, block, 0, stream>>>(start_value, step_value, size, out_data);
}

template decltype(ArangeNullaryKernel<int64_t, GPUContext>) ArangeNullaryKernel;
template decltype(ArangeNullaryKernel<int, GPUContext>) ArangeNullaryKernel;
}  // namespace phi

PD_REGISTER_KERNEL(arange_tensor,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArangeTensorKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_KERNEL(arange,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArangeKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::float16,
                   phi::bfloat16) {}
