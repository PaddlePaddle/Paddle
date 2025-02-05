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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"

#include "paddle/phi/kernels/impl/lu_kernel_impl.h"
#include "paddle/phi/kernels/lu_kernel.h"

namespace phi {

template <typename T, typename Context>
void LUKernel(const Context& dev_ctx,
              const DenseTensor& x,
              bool pivot,
              DenseTensor* out,
              DenseTensor* pivots,
              DenseTensor* infos) {
  PADDLE_ENFORCE_EQ(pivot,
                    true,
                    errors::InvalidArgument(
                        "lu without pivoting is not implemented on the CPU, "
                        "but got pivots=False"));

  *out = Transpose2DTo6D<Context, T>(dev_ctx, x);

  auto outdims = out->dims();
  auto outrank = outdims.size();

  int m = static_cast<int>(outdims[outrank - 1]);
  int n = static_cast<int>(outdims[outrank - 2]);
  int lda = std::max(1, m);

  auto ipiv_dims = common::slice_ddim(outdims, 0, outrank - 1);
  ipiv_dims[outrank - 2] = std::min(m, n);
  pivots->Resize(ipiv_dims);
  dev_ctx.template Alloc<int>(pivots);
  auto ipiv_data = pivots->data<int>();

  auto info_dims = common::slice_ddim(outdims, 0, outrank - 2);
  infos->Resize(info_dims);
  dev_ctx.template Alloc<int>(infos);
  auto info_data = infos->data<int>();

  auto batchsize = product(info_dims);
  batchsize = std::max(static_cast<int>(batchsize), 1);
  dev_ctx.template Alloc<T>(out);
  auto out_data = out->data<T>();
  for (int b = 0; b < batchsize; b++) {
    auto out_data_item = &out_data[b * m * n];
    int* info_data_item = &info_data[b];
    int* ipiv_data_item = &ipiv_data[b * std::min(m, n)];
    phi::funcs::lapackLu<T>(
        m, n, out_data_item, lda, ipiv_data_item, info_data_item);
  }
  *out = Transpose2DTo6D<Context, T>(dev_ctx, *out);
}

}  // namespace phi

PD_REGISTER_KERNEL(lu, CPU, ALL_LAYOUT, phi::LUKernel, float, double) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
