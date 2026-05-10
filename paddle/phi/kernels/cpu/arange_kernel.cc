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

#include "paddle/phi/kernels/arange_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/range_function.h"

namespace phi {

template <typename T, typename Context>
void ArangeFunc(const Context& dev_ctx,
                const T& start_value,
                const T& end_value,
                const T& step_value,
                DenseTensor* out) {
  int64_t size = 0;
  funcs::GetSize(start_value, end_value, step_value, &size);
  out->Resize({size});
  T* out_data = dev_ctx.template Alloc<T>(out);
  T value = start_value;
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = value;
    value += step_value;
  }
}

template <typename T, typename Context>
void ArangeTensorKernel(const Context& dev_ctx,
                        const DenseTensor& start,
                        const DenseTensor& end,
                        const DenseTensor& step,
                        DenseTensor* out) {
  int64_t size = 0;
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  bool any_float = phi::IsFloatingType(start.dtype()) ||
                   phi::IsFloatingType(end.dtype()) ||
                   phi::IsFloatingType(step.dtype());

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
  MPType value = start_value;
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = static_cast<T>(value);
    value += step_value;
  }
}

template <typename T, typename Context>
void ArangeKernel(const Context& dev_ctx,
                  const Scalar& start,
                  const Scalar& end,
                  const Scalar& step,
                  DenseTensor* out) {
  bool any_float = phi::IsFloatingType(start.dtype()) ||
                   phi::IsFloatingType(end.dtype()) ||
                   phi::IsFloatingType(step.dtype());
  int64_t size = 0;
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  if (any_float) {
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
  MPType value = start_value;
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = static_cast<T>(value);
    value += step_value;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(arange_tensor,
                   CPU,
                   ALL_LAYOUT,
                   phi::ArangeTensorKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(
    arange, CPU, ALL_LAYOUT, phi::ArangeKernel, float, double, int, int64_t) {}
