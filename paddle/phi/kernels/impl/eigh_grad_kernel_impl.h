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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/diag_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"
#include "paddle/phi/kernels/math_kernel.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
void EighGradKernel(const Context& dev_ctx,
                    const DenseTensor& out_v,
                    const DenseTensor& out_w,
                    const DenseTensor& dout_v,
                    const DenseTensor& dout_w,
                    DenseTensor* dx) {
  //   using ValueType = phi::dtype::Real<T>;
  //   dev_ctx.template Alloc<T>(dx);

  //   auto& dims = out_w.dims();
  //   const int m = dims[dims.size() - 1];

  //   DenseTensor tW =
  //       phi::TransposeLast2Dim<T>(dev_ctx, phi::Conj<T>(dev_ctx, out_w));
  //   DenseTensor W = phi::Subtract<ValueType>(dev_ctx,
  //                                            phi::funcs::Unsqueeze(out_v,
  //                                            -2),
  //                                            phi::funcs::Unsqueeze(out_v,
  //                                            -1));
  //   DenseTensor result = phi::Matmul<T>(dev_ctx, tW, out_w_grad);
  //   result.Resize(dims);
  //   dev_ctx.template Alloc<T>(&result);

  //   std::vector<int> out_shape = phi::vectorize<int>(dims);
  //   DenseTensor constant;
  //   constant.Resize(phi::make_ddim(out_shape));
  //   dev_ctx.template Alloc<T>(&constant);
  //   phi::funcs::SetConstant<Context, T>()(dev_ctx, &constant, T(0.5));

  //   result = phi::Subtract<T>(
  //       dev_ctx,
  //       result,
  //       phi::Conj<T>(dev_ctx, phi::TransposeLast2Dim<T>(dev_ctx, result)));
  //   result = phi::Multiply<T>(dev_ctx, result, constant);

  //   if(result.type() != W.type()){
  //     W = phi::Cast<T>(dev_ctx, W, result.type());
  //     //   auto x_vector = EigenVector<T>::Flatten(&result);
  //     //   auto y_vector = EigenVector<ValueType>::Flatten(&W);
  //     //   auto out_vector = EigenVector<T>::Flatten(&result);
  //     //   auto& place = *ctx.eigen_device();
  //     //   out_vector.device(place) = x_vector / y_vector;
  //   } else {
  //     result = phi::Divide<T>(dev_ctx, result, W);
  //   }
  //   result = phi::funcs::DiagFill<T, T>(dev_ctx, m, m, m, 0, dout_v, result);
  //   *dx = phi::Matmul<T>(dev_ctx, out_w, phi::Matmul<T>(dev_ctx, result,
  //   tW));
}

}  // namespace phi
