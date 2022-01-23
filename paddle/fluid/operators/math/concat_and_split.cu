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

#include "paddle/pten/kernels/gpu/concat_and_split.h"
namespace paddle {
namespace operators {
namespace math {

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class ConcatFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const std::vector<framework::Tensor>& input, int axis,
                  framework::Tensor* output) {
    std::vector<pten::DenseTensor> pt_input{input.begin(), input.end()};

    pten::ConcatImpl<T, platform::CUDADeviceContext>(context, pt_input, axis,
                                                     output);
  }
};

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class SplitFunctor<platform::CUDADeviceContext, T> {
 public:
  SplitFunctor();
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const std::vector<const framework::Tensor*>& ref_inputs,
                  int axis, std::vector<framework::Tensor*>* outputs) {
    std::vector<const pten::DenseTensor*> pt_ref_inputs{ref_inputs.begin(),
                                                        ref_inputs.end()};
    std::vector<pten::DenseTensor*> pt_outputs{outputs->begin(),
                                               outputs->end()};
    pten::SplitImpl<T, platform::CUDADeviceContext>(
        context, input, pt_ref_inputs, axis, &pt_outputs);
  }
};

#define DEFINE_FUNCTOR(type)                                       \
  template class ConcatFunctor<platform::CUDADeviceContext, type>; \
  template class SplitFunctor<platform::CUDADeviceContext, type>

FOR_ALL_TYPES(DEFINE_FUNCTOR);

}  // namespace math
}  // namespace operators
}  // namespace paddle
