/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/operators/math/softmax_impl.h"

namespace paddle {
namespace operators {
namespace math {

template class SoftmaxFunctor<platform::CPUDeviceContext, float, true>;
template class SoftmaxFunctor<platform::CPUDeviceContext, float, false>;
template class SoftmaxFunctor<platform::CPUDeviceContext, double, true>;
template class SoftmaxFunctor<platform::CPUDeviceContext, double, false>;
template class SoftmaxGradFunctor<platform::CPUDeviceContext, float>;
template class SoftmaxGradFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
