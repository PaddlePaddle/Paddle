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

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/unary.h"

namespace phi {

template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const IntArray& sections,
                 const Scalar& axis,
                 std::vector<DenseTensor*> out);

template <typename T, typename Context>
void SplitWithNumKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        int num,
                        const Scalar& axis,
                        std::vector<DenseTensor*> out);

template <typename T, typename Context>
std::vector<DenseTensor> Split(const Context& dev_ctx,
                               const DenseTensor& x,
                               const IntArray& sections,
                               const Scalar& axis) {
  size_t out_number;
  out_number = sections.GetData().size();

  std::vector<MetaTensor> out_meta;
  std::vector<MetaTensor*> out_meta_ptr;
  out_meta.reserve(out_number);
  out_meta_ptr.reserve(out_number);
  std::vector<DenseTensor> result(out_number);

  for (size_t i = 0; i < out_number; ++i) {
    out_meta.emplace_back(&result[i]);
    out_meta_ptr.push_back(&out_meta.back());
  }
  SplitInferMeta(x, sections, axis, out_meta_ptr);
  std::vector<DenseTensor*> outs;
  outs.reserve(out_meta.size());
  for (size_t i = 0; i < out_meta.size(); ++i) {
    outs.push_back(&result[i]);
  }

  SplitKernel<T, Context>(dev_ctx, x, sections, axis, outs);
  return result;
}

template <typename T, typename Context>
std::vector<DenseTensor> SplitWithNum(const Context& dev_ctx,
                                      const DenseTensor& x,
                                      int num,
                                      const Scalar& axis) {
  size_t out_number = num;

  std::vector<MetaTensor> out_meta;
  std::vector<MetaTensor*> out_meta_ptr;
  out_meta.reserve(out_number);
  out_meta_ptr.reserve(out_number);
  std::vector<DenseTensor> result(out_number);

  for (size_t i = 0; i < out_number; ++i) {
    out_meta.emplace_back(&result[i]);
    out_meta_ptr.push_back(&out_meta.back());
  }
  SplitWithNumInferMeta(x, num, axis, out_meta_ptr);

  std::vector<DenseTensor*> outs;
  outs.reserve(out_meta.size());
  for (size_t i = 0; i < out_meta.size(); ++i) {
    outs.push_back(&result[i]);
  }

  SplitWithNumKernel<T, Context>(dev_ctx, x, num, axis, outs);

  return result;
}

}  // namespace phi
