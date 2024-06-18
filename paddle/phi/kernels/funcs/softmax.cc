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

#include "paddle/phi/kernels/funcs/softmax.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/kernels/funcs/softmax_impl.h"

namespace phi::funcs {

template class SoftmaxFunctor<phi::CPUContext, float>;
template class SoftmaxFunctor<phi::CPUContext, double>;
template class SoftmaxGradFunctor<phi::CPUContext, float>;
template class SoftmaxGradFunctor<phi::CPUContext, double>;

}  // namespace phi::funcs
