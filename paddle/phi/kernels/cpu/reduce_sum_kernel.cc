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

#include <set>

#include "paddle/phi/kernels/reduce_sum_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/reduce.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"

namespace phi {

template <typename T, typename Context>
void SumRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const IntArray& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DataType out_dtype,
                  DenseTensor* out) {
  if (out_dtype == DataType::UNDEFINED && out->dtype() != x.dtype()) {
    out_dtype = out->dtype();
  }
  if (x.numel() == 0) {
    auto x_dims = x.dims();
    std::vector<int> out_dims;
    if (reduce_all) {
      if (keep_dim) {
        out_dims.resize(x_dims.size(), 1);
      } else {
        out_dims = std::vector<int>();
      }
    } else {
      std::set<int> reduce_dims;
      auto dims_vec = dims.GetData();
      for (auto dim : dims_vec) {
        PADDLE_ENFORCE_GE(dim,
                          -x_dims.size(),
                          common::errors::InvalidArgument(
                              "The dimension index is out of range, "
                              "expected index >= %d, but received %d.",
                              -x_dims.size(),
                              dim));
        PADDLE_ENFORCE_LT(dim,
                          x_dims.size(),
                          common::errors::InvalidArgument(
                              "The dimension index is out of range, "
                              "expected index < %d, but received %d.",
                              x_dims.size(),
                              dim));
        if (dim < 0) {
          dim += x_dims.size();
        }
        reduce_dims.insert(dim);
      }
      if (keep_dim) {
        out_dims.resize(x_dims.size());
        for (int i = 0; i < x_dims.size(); ++i) {
          if (reduce_dims.count(i)) {
            out_dims[i] = 1;
          } else {
            out_dims[i] = x_dims[i];
          }
        }
      } else {
        for (int i = 0; i < x_dims.size(); ++i) {
          if (!reduce_dims.count(i)) {
            out_dims.push_back(x_dims[i]);
          }
        }
      }
    }
    out->Resize(phi::make_ddim(out_dims));
    dev_ctx.template Alloc<T>(out);
    FullKernel<T, Context>(
        dev_ctx, out_dims, 0, phi::CppTypeToDataType<T>::Type(), out);
    return;
  }
  phi::Reduce<CPUContext, T, phi::funcs::SumFunctor>(
      dev_ctx, x, reduce_all, dims.GetData(), keep_dim, out_dtype, out);
}

}  // namespace phi

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(sum_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::SumRawKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int16_t,
                   int8_t,
                   uint8_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
