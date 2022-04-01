/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LTCENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS TS" BASTS,
WTTHOUT WARRANTTES OR CONDTTTONS OF ANY KTND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/sparse/sparse_addcsr_kernel.h"
//#include "paddle/phi"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {
namespace sparse {

//template <typename T, typename Context>
//void AddCsrKernel(const Context& dev_ctx,
//                  const SparseCsrTensor& x,
//                  const SparseCsrTensor& y,
//                  SparseCsrTensor* out) {
//
//       const I n_row, const I n_col,
//       const I Ap[], const I Aj[], const T Ax[],
//       const I Bp[], const I Bj[], const T Bx[],
//             I Cp[],       I Cj[],       T2 Cx[],
//       const binary_op& op)
//  //Method that works for duplicate and/or unsorted indices
//  n_row=
//
//
//
//  std::vector<T> next(n_col,-1);
//  std::vector<T> A_row(n_col, 0);
//  std::vector<T> B_row(n_col, 0);
//
//  T nnz = 0;
//  Cp[0] = 0;
//
//  for (T i = 0; i < n_row; i++) {
//    T head = -2;
//    T length = 0;
//
//    // add a row of A to A_row
//    T i_start = Ap[i];
//    T i_end = Ap[i + 1];
//    for (T jj = i_start; jj < i_end; jj++) {
//      T j = Aj[jj];
//
//      A_row[j] += Ax[jj];
//
//      if (next[j] == -1) {
//        next[j] = head;
//        head = j;
//        length++;
//      }
//    }
//
//    // add a row of B to B_row
//    i_start = Bp[i];
//    i_end = Bp[i + 1];
//    for (T jj = i_start; jj < i_end; jj++) {
//      T j = Bj[jj];
//
//      B_row[j] += Bx[jj];
//
//      if (next[j] == -1) {
//        next[j] = head;
//        head = j;
//        length++;
//      }
//    }
//
//    // scan through columns where A or B has
//    // contributed a non-zero entry
//    for (T jj = 0; jj < length; jj++) {
//      T result = op(A_row[head], B_row[head]);
//
//      if (result != 0) {
//        Cj[nnz] = head;
//        Cx[nnz] = result;
//        nnz++;
//      }
//
//      T temp = head;
//      head = next[head];
//
//      next[temp] = -1;
//      A_row[temp] = 0;
//      B_row[temp] = 0;
//    }
//
//    Cp[i + 1] = nnz;
//  }
//}

}  // namespace sparse
}  // namespace phi

//PD_REGISTER_KERNEL(sparse_add_csr,
//                   CPU,
//                   ALL_LAYOUT,
//                   phi::sparse::AddCsrKernel,
//                   float,
//                   double,
//                   int,
//                   int64_t) {
//  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
//}
