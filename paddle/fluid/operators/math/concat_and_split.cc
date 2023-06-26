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
#include "paddle/fluid/platform/device_context.h"

#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace operators {
namespace math {

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class ConcatFunctor<phi::CPUContext, T> {
 public:
  void operator()(const phi::CPUContext& context,
                  const std::vector<phi::DenseTensor>& input,
                  int axis,
                  phi::DenseTensor* output) {
    phi::funcs::ConcatFunctor<phi::CPUContext, T> functor;
    functor(context, input, axis, output);
  }
};

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class SplitFunctor<phi::CPUContext, T> {
 public:
  void operator()(const phi::CPUContext& context,
                  const phi::DenseTensor& input,
                  const std::vector<const phi::DenseTensor*>& ref_inputs,
                  const int axis,
                  std::vector<phi::DenseTensor*>* outputs) {
    phi::funcs::SplitFunctor<phi::CPUContext, T> functor;
    functor(context, input, ref_inputs, axis, outputs);
  }
};

#ifdef PADDLE_WITH_XPU
/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class ConcatFunctor<phi::XPUContext, T> {
 public:
  void operator()(const phi::XPUContext& context,
                  const std::vector<phi::DenseTensor>& input,
                  int axis,
                  phi::DenseTensor* output) {
    phi::funcs::ConcatFunctor<phi::XPUContext, T> functor;
    functor(context, input, axis, output);
  }
};

template <typename T>
class SplitFunctor<phi::XPUContext, T> {
 public:
  void operator()(const phi::XPUContext& context,
                  const phi::DenseTensor& input,
                  const std::vector<const phi::DenseTensor*>& ref_inputs,
                  const int axis,
                  std::vector<phi::DenseTensor*>* outputs) {
    phi::funcs::SplitFunctor<phi::XPUContext, T> functor;
    functor(context, input, ref_inputs, axis, outputs);
  }
};
#endif

#define DEFINE_FUNCTOR(type)                           \
  template class ConcatFunctor<phi::CPUContext, type>; \
  template class SplitFunctor<phi::CPUContext, type>;

FOR_ALL_TYPES(DEFINE_FUNCTOR);

#ifdef PADDLE_WITH_XPU
#define DEFINE_XPU_FUNCTOR(type)                       \
  template class ConcatFunctor<phi::XPUContext, type>; \
  template class SplitFunctor<phi::XPUContext, type>;

DEFINE_XPU_FUNCTOR(float)
DEFINE_XPU_FUNCTOR(phi::dtype::float16)
DEFINE_XPU_FUNCTOR(int32_t)
DEFINE_XPU_FUNCTOR(int64_t)
DEFINE_XPU_FUNCTOR(uint8_t)
#endif
}  // namespace math
}  // namespace operators
}  // namespace paddle
