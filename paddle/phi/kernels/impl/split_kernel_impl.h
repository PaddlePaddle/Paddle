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
#include "paddle/phi/kernels/split_kernel.h"

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/strided_memcpy.h"

namespace phi {
template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const IntArray& sections UNUSED,
                 const Scalar& axis_scalar,
                 std::vector<DenseTensor*> outs) {
  std::vector<const DenseTensor*> shape_refer;
  for (size_t j = 0; j < outs.size(); ++j) {
    dev_ctx.template Alloc<T>(outs[j]);
    shape_refer.emplace_back(outs[j]);
  }

  int axis = axis_scalar.to<int>();
  // Sometimes direct copies will be faster, this maybe need deeply analysis.
  if (axis == 0 && outs.size() < 10) {
    phi::funcs::StridedMemcpyWithAxis0<T, Context>(
        dev_ctx, x, shape_refer, &outs);
  } else {
    phi::funcs::SplitFunctor<Context, T> functor;
    functor(dev_ctx, x, shape_refer, axis, &outs);
  }
}

template <typename T, typename Context>
void SplitWithNumKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        int num,
                        const Scalar& axis_scalar,
                        std::vector<DenseTensor*> outs) {
  int axis_value = axis_scalar.to<int>();
  int64_t input_axis_dim = x.dims().at(axis_value);
  std::vector<int64_t> sections_vec;
  if (num > input_axis_dim) {
    // if num > input_axis_dim, take num = input_axis_dim, the dim after split
    // is 1.
    num = input_axis_dim;
  }

  int64_t last_split_size = 0;
  if (input_axis_dim % num > 0) {
    // increase the processing of non-divisible numbers.
    last_split_size = input_axis_dim % num;
    num = num - 1;
    input_axis_dim = input_axis_dim - last_split_size;
  }

  for (int i = 0; i < num; ++i) {
    sections_vec.push_back(input_axis_dim / num);
  }
  if (last_split_size > 0) {
    // place the remainder at the end of Tensor.
    sections_vec.push_back(last_split_size);
  }
  IntArray sections(sections_vec);
  SplitKernel<T, Context>(dev_ctx, x, sections, axis_scalar, outs);
}

}  // namespace phi
