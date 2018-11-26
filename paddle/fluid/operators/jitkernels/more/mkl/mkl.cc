/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "paddle/fluid/operators/jitkernels/more/mkl/mkl.h"
#include "paddle/fluid/operators/jitkernels/registry.h"
#include "paddle/fluid/platform/dynload/mklml.h"

namespace paddle {
namespace operators {
namespace jitkernels {
namespace more {
namespace mkl {

template <>
void VMul<float>(const float* x, const float* y, float* z, int n) {
  platform::dynload::vsMul(n, x, y, z);
}

template <>
void VMul<double>(const double* x, const double* y, double* z, int n) {
  platform::dynload::vdMul(n, x, y, z);
}

}  // namespace mkl
}  // namespace more
}  // namespace jitkernels
}  // namespace operators
}  // namespace paddle

namespace mkl = paddle::operators::jitkernels::more::mkl;

REGISTER_JITKERNEL_MORE(vmul, mkl, mkl::VMulKernel<float>,
                        mkl::VMulKernel<double>);
