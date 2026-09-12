// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_XPU_FFT
#include <string>
#include <vector>

#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fft_kernel.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/fft.h"
#include "paddle/phi/kernels/funcs/fft_fill_conj.h"
#include "paddle/phi/kernels/funcs/fft_fill_conj_xpu.h"

namespace {

// XPU FFT library requires axis size > 8 for correct execution.
// For small axis sizes, fall back to CPU FFT implementation.
bool HasSmallFFTAxis(const phi::DenseTensor& x,
                     const std::vector<int64_t>& axes) {
  for (auto axis : axes) {
    if (x.dims()[axis] <= 8) {
      return true;
    }
  }
  return false;
}

}  // namespace

namespace phi {
template <typename T, typename Context>
void FFTC2CKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& axes,
                  const std::string& normalization,
                  bool forward,
                  DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (x.numel() == 0) {
    Full<T, Context>(dev_ctx, out->dims(), 0, out);
    return;
  }
  const auto norm_type = funcs::get_norm_from_string(normalization, forward);

  if (HasSmallFFTAxis(x, axes)) {
    phi::CPUPlace cpu_place;
    phi::CPUContext cpu_ctx;
    DenseTensor x_cpu;
    x_cpu.Resize(x.dims());
    auto* x_cpu_ptr = x_cpu.mutable_data<T>(cpu_place);
    memory_utils::Copy(cpu_place,
                       x_cpu_ptr,
                       dev_ctx.GetPlace(),
                       x.data<T>(),
                       x.numel() * sizeof(T));
    DenseTensor out_cpu;
    out_cpu.Resize(out->dims());
    auto* out_cpu_ptr = out_cpu.mutable_data<T>(cpu_place);
    funcs::FFTC2CFunctor<phi::CPUContext, T, T> fft_c2c_func;
    fft_c2c_func(cpu_ctx, x_cpu, &out_cpu, axes, norm_type, forward);
    memory_utils::Copy(dev_ctx.GetPlace(),
                       out->data<T>(),
                       cpu_place,
                       out_cpu_ptr,
                       out->numel() * sizeof(T));
    return;
  }

  funcs::FFTC2CFunctor<Context, T, T> fft_c2c_func;
  fft_c2c_func(dev_ctx, x, out, axes, norm_type, forward);
}

template <typename T, typename Context>
void FFTC2RKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& axes,
                  const std::string& normalization,
                  bool forward,
                  int64_t last_dim_size UNUSED,
                  DenseTensor* out) {
  using R = typename T::value_type;  // get real type
  dev_ctx.template Alloc<R>(out);
  if (x.numel() == 0) {
    Full<R, Context>(dev_ctx, out->dims(), 0, out);
    return;
  }
  const auto norm_type = funcs::get_norm_from_string(normalization, forward);

  if (HasSmallFFTAxis(x, axes)) {
    phi::CPUPlace cpu_place;
    phi::CPUContext cpu_ctx;
    DenseTensor x_cpu;
    x_cpu.Resize(x.dims());
    auto* x_cpu_ptr = x_cpu.mutable_data<T>(cpu_place);
    memory_utils::Copy(cpu_place,
                       x_cpu_ptr,
                       dev_ctx.GetPlace(),
                       x.data<T>(),
                       x.numel() * sizeof(T));
    DenseTensor out_cpu;
    out_cpu.Resize(out->dims());
    auto* out_cpu_ptr = out_cpu.mutable_data<R>(cpu_place);
    funcs::FFTC2RFunctor<phi::CPUContext, T, R> fft_c2r_func;
    fft_c2r_func(cpu_ctx, x_cpu, &out_cpu, axes, norm_type, forward);
    memory_utils::Copy(dev_ctx.GetPlace(),
                       out->data<R>(),
                       cpu_place,
                       out_cpu_ptr,
                       out->numel() * sizeof(R));
    return;
  }

  funcs::FFTC2RFunctor<Context, T, R> fft_c2r_func;
  fft_c2r_func(dev_ctx, x, out, axes, norm_type, forward);
}

template <typename T, typename Context>
void FFTR2CKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& axes,
                  const std::string& normalization,
                  bool forward,
                  bool onesided,
                  DenseTensor* out) {
  using C = phi::dtype::complex<T>;
  dev_ctx.template Alloc<C>(out);
  if (x.numel() == 0) {
    Full<C, Context>(dev_ctx, out->dims(), 0, out);
    return;
  }
  auto norm_type = funcs::get_norm_from_string(normalization, forward);

  if (HasSmallFFTAxis(x, axes)) {
    phi::CPUPlace cpu_place;
    phi::CPUContext cpu_ctx;
    DenseTensor x_cpu;
    x_cpu.Resize(x.dims());
    auto* x_cpu_ptr = x_cpu.mutable_data<T>(cpu_place);
    memory_utils::Copy(cpu_place,
                       x_cpu_ptr,
                       dev_ctx.GetPlace(),
                       x.data<T>(),
                       x.numel() * sizeof(T));
    DenseTensor out_cpu;
    out_cpu.Resize(out->dims());
    auto* out_cpu_ptr = out_cpu.mutable_data<C>(cpu_place);
    funcs::FFTR2CFunctor<phi::CPUContext, T, C> fft_r2c_func;

    if (onesided) {
      fft_r2c_func(cpu_ctx, x_cpu, &out_cpu, axes, norm_type, forward);
    } else {
      DDim onesided_out_shape = x.dims();
      const int64_t last_fft_axis = axes.back();
      const int64_t onesided_last_axis_size =
          out->dims().at(last_fft_axis) / 2 + 1;
      onesided_out_shape[last_fft_axis] = onesided_last_axis_size;
      DenseTensor onesided_out_cpu;
      onesided_out_cpu.Resize(onesided_out_shape);
      onesided_out_cpu.mutable_data<C>(cpu_place);
      fft_r2c_func(cpu_ctx, x_cpu, &onesided_out_cpu, axes, norm_type, forward);

      // Use FFTFillConjFunctor directly for CPU (avoid XPU template conflict)
      std::vector<int64_t> src_strides_v =
          vectorize<int64_t>(common::stride(onesided_out_cpu.dims()));
      std::vector<int64_t> dst_strides_v =
          vectorize<int64_t>(common::stride(out_cpu.dims()));
      std::vector<int64_t> dst_shape_v = vectorize<int64_t>(out_cpu.dims());
      auto _is_fft_axis = std::make_unique<bool[]>(out_cpu.dims().size());
      for (const auto i : axes) {
        _is_fft_axis[i] = true;
      }
      funcs::FFTFillConjFunctor<C> fill_conj_functor(
          onesided_out_cpu.data<C>(),
          out_cpu_ptr,
          src_strides_v.data(),
          dst_strides_v.data(),
          dst_shape_v.data(),
          _is_fft_axis.get(),
          static_cast<int64_t>(last_fft_axis),
          static_cast<int64_t>(onesided_last_axis_size),
          static_cast<int64_t>(out_cpu.dims().size()));
      funcs::ForRange<phi::CPUContext> for_range(cpu_ctx, out_cpu.numel());
      for_range(fill_conj_functor);
    }

    memory_utils::Copy(dev_ctx.GetPlace(),
                       out->data<C>(),
                       cpu_place,
                       out_cpu_ptr,
                       out->numel() * sizeof(C));
    return;
  }

  funcs::FFTR2CFunctor<Context, T, C> fft_r2c_func;

  if (onesided) {
    fft_r2c_func(dev_ctx, x, out, axes, norm_type, forward);
  } else {
    DDim onesided_out_shape = x.dims();
    const int64_t last_fft_axis = axes.back();
    const int64_t onesided_last_axis_size =
        out->dims().at(last_fft_axis) / 2 + 1;
    onesided_out_shape[last_fft_axis] = onesided_last_axis_size;
    DenseTensor onesided_out =
        Empty<C, Context>(dev_ctx, vectorize(onesided_out_shape));
    fft_r2c_func(dev_ctx, x, &onesided_out, axes, norm_type, forward);
    funcs::FFTFillConj<Context, C>(dev_ctx, &onesided_out, out, axes);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    fft_c2c, XPU, ALL_LAYOUT, phi::FFTC2CKernel, phi::complex64) {}
PD_REGISTER_KERNEL(
    fft_c2r, XPU, ALL_LAYOUT, phi::FFTC2RKernel, phi::complex64) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
PD_REGISTER_KERNEL(fft_r2c, XPU, ALL_LAYOUT, phi::FFTR2CKernel, float) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToComplex(kernel_key.dtype()));
}
#endif
