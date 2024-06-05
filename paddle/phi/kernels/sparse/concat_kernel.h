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

#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"
#include "paddle/phi/kernels/funcs/strided_memcpy.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi {
namespace sparse {
// This Functor is based on DenseTensor's concat
template <typename T, typename Context>
void ConcatFunctor(const Context& context,
                   const std::vector<phi::DenseTensor>& x,
                   int axis,
                   phi::DenseTensor* out) {
  // Sometimes direct copies will be faster, this maybe need deeply analysis.
  if (axis == 0 && x.size() < 10) {
    size_t output_offset = 0;
    for (auto in : x) {
      if (in.numel() == 0UL) {
        continue;
      }
      auto in_stride = common::stride_numel(in.dims());
      auto out_stride = common::stride_numel(out->dims());
      phi::funcs::StridedNumelCopyWithAxis<T, Context>(
          context,
          axis,
          out->data<T>() + output_offset,
          out_stride,
          in.data<T>(),
          in_stride,
          in_stride[axis]);
      output_offset += in_stride[axis];
    }
  } else {
    // TODO(bapijun) use a std::vector<DenseTensor *> to optimize the
    // concat_functor
    phi::funcs::ConcatFunctor<Context, T> functor;
    functor(context, x, axis, out);
  }
}

template <typename T, typename Context>
void ConcatCooKernel(const Context& dev_ctx,
                     const std::vector<const SparseCooTensor*>& x,
                     const Scalar& axis_scalar,
                     SparseCooTensor* out);

template <typename T, typename Context>
void ConcatCsrKernel(const Context& dev_ctx,
                     const std::vector<const SparseCsrTensor*>& x,
                     const Scalar& axis_scalar,
                     SparseCsrTensor* out);

}  // namespace sparse
}  // namespace phi
