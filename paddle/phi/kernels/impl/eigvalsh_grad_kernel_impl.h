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

#pragma once

#pragma once

#include "paddle/phi/kernels/eigvalsh_grad_kernel.h"

#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
void EigvalshGradKernel(const Context& dev_ctx,
                        const DenseTensor& out_v,
                        const DenseTensor& out_w_grad,
                        const std::string& uplo UNUSED,
                        bool is_test UNUSED,
                        DenseTensor* x_grad) {
  auto tV = phi::TransposeLast2Dim<T>(dev_ctx, phi::Conj<T>(dev_ctx, out_v));

  x_grad->Resize(out_v.dims());
  dev_ctx.template Alloc<T>(x_grad);

  auto output_v_vector = EigenVector<T>::Flatten(out_v);
  auto output_w_grad_vector =
      EigenVector<phi::dtype::Real<T>>::Flatten(out_w_grad);
  auto result_vector = EigenVector<T>::Flatten(*x_grad);
  auto& place = *dev_ctx.eigen_device();
  std::vector<int> broadcast_factor;
  broadcast_factor.push_back(out_v.dims().at(out_v.dims().size() - 1));
  result_vector.device(place) =
      output_v_vector * output_w_grad_vector.broadcast(broadcast_factor);

  *x_grad = phi::Matmul<T>(dev_ctx, *x_grad, tV);
}

}  // namespace phi
