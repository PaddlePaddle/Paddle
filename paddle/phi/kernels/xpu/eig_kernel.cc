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

#include "paddle/phi/kernels/eig_kernel.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/eig.h"

namespace phi {

template <typename T, typename Context>
void EigKernel(const Context& dev_ctx,
               const DenseTensor& x,
               DenseTensor* out_w,
               DenseTensor* out_v) {
  dev_ctx.template Alloc<phi::dtype::Complex<T>>(out_w);
  dev_ctx.template Alloc<phi::dtype::Complex<T>>(out_v);

  if (x.numel() == 0) {
    return;
  }

  auto cpu_place = CPUPlace();
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* cpu_ctx = static_cast<phi::CPUContext*>(pool.Get(cpu_place));

  // prepare cpu Tensor here, since magma requires output on cpu
  DenseTensor out_w_cpu, out_v_cpu;
  out_w_cpu.Resize(out_w->dims());
  (*cpu_ctx).template Alloc<phi::dtype::Complex<T>>(&out_w_cpu);
  out_v_cpu.Resize(x.dims());
  (*cpu_ctx).template Alloc<phi::dtype::Complex<T>>(&out_v_cpu);

  if (!IsComplexType(x.dtype())) {
    // output still be complex though input is real
    int batch_count = BatchCount(x);
    int order = static_cast<int>(x.dims(-1));

    DenseTensor real_w_cpu, real_v_cpu;

    std::vector<int64_t> real_w_dim = common::vectorize<int64_t>(out_w->dims());
    real_w_dim.back() *= 2;
    real_w_cpu.Resize(common::make_ddim(real_w_dim));
    (*cpu_ctx).template Alloc<phi::dtype::Real<T>>(&real_w_cpu);
    real_v_cpu.Resize(x.dims());
    (*cpu_ctx).template Alloc<phi::dtype::Real<T>>(&real_v_cpu);

    phi::ApplyEigKernelMagma<phi::dtype::Real<T>, Context>(
        dev_ctx, x, &real_w_cpu, &real_v_cpu);

    // 1. extract real part & imag part from real_w_cpu
    DenseTensor real_part_cpu = funcs::Slice<phi::dtype::Real<T>>(
        (*cpu_ctx), real_w_cpu, {-1}, {0}, {order});
    DenseTensor imag_part_cpu = funcs::Slice<phi::dtype::Real<T>>(
        (*cpu_ctx), real_w_cpu, {-1}, {order}, {order * 2});

    // 2. construct complex values
    auto* real_part_data = real_part_cpu.data<phi::dtype::Real<T>>();
    auto* imag_part_data = imag_part_cpu.data<phi::dtype::Real<T>>();
    int64_t out_w_numel = static_cast<int64_t>(out_w->numel());

    funcs::ForRange<phi::CPUContext> for_range((*cpu_ctx), out_w_numel);
    funcs::RealImagToComplexFunctor<phi::dtype::Complex<T>> functor(
        real_part_data,
        imag_part_data,
        out_w_cpu.data<phi::dtype::Complex<T>>(),
        out_w_numel);
    for_range(functor);

    // 3. construct complex vectors
    DenseTensor real_v_trans_cpu =
        TransposeLast2Dim<phi::dtype::Real<T>, phi::CPUContext>((*cpu_ctx),
                                                                real_v_cpu);
    DenseTensor out_v_trans_cpu;
    out_v_trans_cpu.Resize(x.dims());
    (*cpu_ctx).template Alloc<phi::dtype::Complex<T>>(&out_v_trans_cpu);

    phi::ConstructComplexVectors<phi::dtype::Real<T>,
                                 phi::dtype::Complex<T>,
                                 phi::CPUContext>(&out_v_trans_cpu,
                                                  out_w_cpu,
                                                  real_v_trans_cpu,
                                                  (*cpu_ctx),
                                                  batch_count,
                                                  order);

    TransposeTwoAxis<phi::dtype::Complex<T>, phi::CPUContext>(
        out_v_trans_cpu,
        &out_v_cpu,
        x.dims().size() - 1,
        x.dims().size() - 2,
        (*cpu_ctx));

  } else {
    phi::ApplyEigKernelMagma<T, Context>(dev_ctx, x, &out_w_cpu, &out_v_cpu);
  }

  // copy result from cpu to xpu tensor
  Copy(dev_ctx, out_w_cpu, phi::XPUPlace(), false, out_w);
  Copy(dev_ctx, out_v_cpu, phi::XPUPlace(), false, out_v);
}

}  // namespace phi

#ifdef PADDLE_WITH_MAGMA
PD_REGISTER_KERNEL(
    eig, XPU, ALL_LAYOUT, phi::EigKernel, float, phi::complex64) {
  if (kernel_key.dtype() == phi::DataType::FLOAT32 ||
      kernel_key.dtype() == phi::DataType::FLOAT64) {
    kernel->OutputAt(0).SetDataType(phi::dtype::ToComplex(kernel_key.dtype()));
    kernel->OutputAt(1).SetDataType(phi::dtype::ToComplex(kernel_key.dtype()));
  }
}
#endif
