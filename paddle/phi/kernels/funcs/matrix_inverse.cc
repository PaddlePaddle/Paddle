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

#include "paddle/phi/kernels/funcs/matrix_inverse.h"

#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi::funcs {

template <typename Context, typename T>
void MatrixInverseFunctor<Context, T>::operator()(const Context& dev_ctx,
                                                  const DenseTensor& a,
                                                  DenseTensor* a_inv) {
  ComputeInverseEigen<Context, T>(dev_ctx, a, a_inv);
}

template class MatrixInverseFunctor<CPUContext, float>;
template class MatrixInverseFunctor<CPUContext, double>;
template class MatrixInverseFunctor<CPUContext, phi::dtype::complex<float>>;
template class MatrixInverseFunctor<CPUContext, phi::dtype::complex<double>>;

}  // namespace phi::funcs
