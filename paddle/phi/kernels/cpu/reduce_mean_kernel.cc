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

#include "paddle/phi/kernels/reduce_mean_kernel.h"

#include "paddle/common/flags.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/reduce.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/cascade_sum.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

namespace phi {
namespace {

template <typename T>
struct MeanDivisor {
  static void Apply(T* data, int64_t numel, int64_t n) {
    const T divisor = static_cast<T>(n);
    for (int64_t i = 0; i < numel; ++i) data[i] /= divisor;
  }
};

// c10::complex's operator/ degenerates to a multiplication by the reciprocal
// when the divisor is real (`scl = 1 / c`, then `real * scl`, `imag * scl`), so
// dividing each component by n directly would not be bit-identical.
template <typename R>
struct MeanDivisor<dtype::complex<R>> {
  static void Apply(dtype::complex<R>* data, int64_t numel, int64_t n) {
    const R scale = static_cast<R>(1) / static_cast<R>(n);
    for (int64_t i = 0; i < numel; ++i) {
      data[i] = dtype::complex<R>(data[i].real * scale, data[i].imag * scale);
    }
  }
};

// torch computes the CPU mean as `sum_out(...).div_(dim_prod)`, there is no
// dedicated CPU mean kernel (mean_stub is registered as nullptr, see
// TORCH_IMPL_FUNC(mean_out) in aten/src/ATen/native/ReduceOps.cpp).
template <typename T, typename Context>
void CascadeMean(const Context& dev_ctx,
                 const DenseTensor& x,
                 const std::vector<int64_t>& axes,
                 DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  funcs::TorchCompatibleReduceSum<T>(x.data<T>(),
                                     common::vectorize(x.dims()),
                                     common::vectorize(x.strides()),
                                     axes,
                                     out->data<T>(),
                                     out->numel());
  MeanDivisor<T>::Apply(out->data<T>(), out->numel(), x.numel() / out->numel());
}

}  // namespace

template <typename T, typename Context>
void MeanRawKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const IntArray& dims,
                   bool keep_dim,
                   bool reduce_all,
                   DenseTensor* out) {
  if (x.numel() == 0) {
    Full<T, Context>(dev_ctx, out->dims(), NAN, out);
    return;
  }

  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                std::is_same_v<T, complex64> || std::is_same_v<T, complex128>) {
    if (FLAGS_use_accuracy_compatible_kernel) {
      CascadeMean<T, Context>(
          dev_ctx,
          x,
          funcs::NormalizeReduceAxes(x.dims(), dims.GetData(), reduce_all),
          out);
      return;
    }
  }

  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  auto out_dtype = x.dtype();
  Reduce<CPUContext, T, funcs::MeanFunctor>(
      dev_ctx, x, reduce_all, dims.GetData(), keep_dim, out_dtype, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(mean_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::MeanRawKernel,
                   float,
                   double,
                   bool,
                   int,
                   int64_t,
                   phi::complex64,
                   phi::complex128) {}
