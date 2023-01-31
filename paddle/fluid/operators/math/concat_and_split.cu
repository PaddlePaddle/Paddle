/* Copyright (c) 2018 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
namespace paddle {
namespace operators {
namespace math {

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class ConcatFunctor<phi::GPUContext, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const std::vector<phi::DenseTensor>& input,
                  int axis,
                  phi::DenseTensor* output) {
    phi::funcs::ConcatFunctor<phi::GPUContext, T> functor;
    functor(context, input, axis, output);
  }
};

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class SplitFunctor<phi::GPUContext, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const phi::DenseTensor& input,
                  const std::vector<const phi::DenseTensor*>& ref_inputs,
                  int axis,
                  std::vector<phi::DenseTensor*>* outputs) {
    phi::funcs::SplitFunctor<phi::GPUContext, T> functor;
    functor(context, input, ref_inputs, axis, outputs);
  }
};

#define DEFINE_FUNCTOR(type)                           \
  template class ConcatFunctor<phi::GPUContext, type>; \
  template class SplitFunctor<phi::GPUContext, type>

FOR_ALL_TYPES(DEFINE_FUNCTOR);

}  // namespace math
}  // namespace operators
}  // namespace paddle
