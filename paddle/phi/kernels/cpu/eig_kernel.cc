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

#include "paddle/phi/kernels/eig_kernel.h"
#include "paddle/phi/kernels/cpu/eig.h"

#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void EigKernel(const Context& dev_ctx,
               const DenseTensor& x,
               DenseTensor* out_w,
               DenseTensor* out_v) {
  PADDLE_ENFORCE_GT(
      x.numel(),
      0,
      errors::InvalidArgument("EigKernel input tensor is empty."));
  if (!IsComplexType(x.dtype())) {
    dev_ctx.template Alloc<phi::dtype::Complex<T>>(out_w);
    dev_ctx.template Alloc<phi::dtype::Complex<T>>(out_v);

    int batch_count = BatchCount(x);
    int order = static_cast<int>(x.dims()[x.dims().size() - 1]);

    PADDLE_ENFORCE_LT(0,
                      order,
                      errors::InvalidArgument(
                          "The order of Input(X) should be greater than 0."));

    DenseTensor real_w;
    DenseTensor real_v;

    // double the size of real_w, the first half stores the real part,
    // the next half stores the imag part
    std::vector<int> origin_dim = common::vectorize<int>(out_w->dims());
    int last_item = origin_dim.back();
    origin_dim.pop_back();
    origin_dim.push_back(last_item * 2);

    phi::DDim big_dim = common::make_ddim(origin_dim);

    real_w.Resize(big_dim);
    dev_ctx.template Alloc<phi::dtype::Real<T>>(&real_w);
    real_v.Resize(x.dims());
    dev_ctx.template Alloc<phi::dtype::Real<T>>(&real_v);

    phi::ApplyEigKernel<phi::dtype::Real<T>, Context>(
        x, &real_w, &real_v, dev_ctx);

    // 1. extract real part & imag part from real_w
    DenseTensor real_part =
        phi::funcs::Slice<T>(dev_ctx, real_w, {-1}, {0}, {order});
    DenseTensor imag_part =
        phi::funcs::Slice<T>(dev_ctx, real_w, {-1}, {order}, {order * 2});

    // 2. construct complex values
    auto* real_part_data = real_part.data<phi::dtype::Real<T>>();
    auto* imag_part_data = imag_part.data<phi::dtype::Real<T>>();
    int out_w_numel = static_cast<int>(out_w->numel());

    phi::funcs::ForRange<Context> for_range(dev_ctx, out_w_numel);
    phi::funcs::RealImagToComplexFunctor<phi::dtype::Complex<T>> functor(
        real_part_data,
        imag_part_data,
        dev_ctx.template Alloc<phi::dtype::Complex<T>>(out_w),
        out_w_numel);

    for_range(functor);

    // 3. construct complex vectors
    DenseTensor real_vector_trans = phi::TransposeLast2Dim<T>(dev_ctx, real_v);
    DenseTensor out_v_trans;
    out_v_trans.Resize(x.dims());
    dev_ctx.template Alloc<phi::dtype::Complex<T>>(&out_v_trans);
    phi::ConstructComplexVectors<phi::dtype::Real<T>,
                                 phi::dtype::Complex<T>,
                                 Context>(
        &out_v_trans, *out_w, real_vector_trans, dev_ctx, batch_count, order);
    TransposeTwoAxis<phi::dtype::Complex<T>, Context>(
        out_v_trans, out_v, x.dims().size() - 1, x.dims().size() - 2, dev_ctx);
  } else {
    dev_ctx.template Alloc<T>(out_w);
    dev_ctx.template Alloc<T>(out_v);

    phi::ApplyEigKernel<T, Context>(x, out_w, out_v, dev_ctx);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(eig,
                   CPU,
                   ALL_LAYOUT,
                   phi::EigKernel,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  if (kernel_key.dtype() == phi::DataType::FLOAT32 ||
      kernel_key.dtype() == phi::DataType::FLOAT64) {
    kernel->OutputAt(0).SetDataType(phi::dtype::ToComplex(kernel_key.dtype()));
    kernel->OutputAt(1).SetDataType(phi::dtype::ToComplex(kernel_key.dtype()));
  }
}
