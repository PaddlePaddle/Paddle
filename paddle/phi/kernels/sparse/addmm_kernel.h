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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

namespace phi {
namespace sparse {

// TODO(zhouwei25): implement " COO + COO @ COO -> COO"
template <typename T, typename Context>
void AddmmCooCooKernel(const Context& dev_ctx,
                       const SparseCooTensor& input,
                       const SparseCooTensor& x,
                       const SparseCooTensor& y,
<<<<<<< HEAD
                       float beta,
                       float alpha,
=======
                       float alpha,
                       float beta,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                       SparseCooTensor* out);

/* DENSE + COO @ DENSE -> DENSE */
template <typename T, typename Context>
void AddmmCooDenseKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const SparseCooTensor& x,
                         const DenseTensor& y,
<<<<<<< HEAD
                         float beta,
                         float alpha,
=======
                         float alpha,
                         float beta,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                         DenseTensor* out);

// TODO(zhouwei25): implement " CSR + CSR @ CSR -> CSR"
template <typename T, typename Context>
void AddmmCsrCsrKernel(const Context& dev_ctx,
                       const SparseCsrTensor& input,
                       const SparseCsrTensor& x,
                       const SparseCsrTensor& y,
<<<<<<< HEAD
                       float beta,
                       float alpha,
=======
                       float alpha,
                       float beta,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                       SparseCsrTensor* out);

/* DENSE + CSR @ DENSE -> DENSE */
template <typename T, typename Context>
void AddmmCsrDenseKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const SparseCsrTensor& x,
                         const DenseTensor& y,
<<<<<<< HEAD
                         float beta,
                         float alpha,
=======
                         float alpha,
                         float beta,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                         DenseTensor* out);

}  // namespace sparse
}  // namespace phi
