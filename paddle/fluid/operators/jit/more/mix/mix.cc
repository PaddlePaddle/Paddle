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

#include "paddle/fluid/operators/jit/more/mix/mix.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/jit/registry.h"
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {
namespace jit {
namespace more {
namespace mix {

template <typename T>
void VSigmoid(const T* x, T* y, int n) {
  const float min = SIGMOID_THRESHOLD_MIN;
  const float max = SIGMOID_THRESHOLD_MAX;
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    y[i] = static_cast<T>(0) - y[i];
  }
  auto compute = Get<KernelType::vexp, XYNTuples<T>, platform::CPUPlace>(n);
  compute(y, y, n);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(1) / (static_cast<T>(1) + y[i]);
  }
}

template <typename T>
void VTanh(const T* x, T* y, int n) {
  const T a = 2, b = -1;
  auto compute_scal = Get<vscal, AXYNTuples<T>, platform::CPUPlace>(n);
  auto compute_addbias = Get<vaddbias, AXYNTuples<T>, platform::CPUPlace>(n);
  auto compute_sigmoid = Get<vsigmoid, XYNTuples<T>, platform::CPUPlace>(n);
  compute_scal(&a, x, y, n);
  compute_sigmoid(y, y, n);
  compute_scal(&a, y, y, n);
  compute_addbias(&b, y, y, n);
}

template <>
bool VSigmoidKernel<float>::UseMe(int d) const {
  return true;
}

template <>
bool VTanhKernel<float>::UseMe(int d) const {
  return true;
}

#define AWALYS_USE_ME_WITH_DOUBLE(func)           \
  template <>                                     \
  bool func##Kernel<double>::UseMe(int d) const { \
    return true;                                  \
  }

AWALYS_USE_ME_WITH_DOUBLE(VSigmoid);
AWALYS_USE_ME_WITH_DOUBLE(VTanh);

#undef AWALYS_USE_ME_WITH_DOUBLE

}  // namespace mix
}  // namespace more
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace mix = paddle::operators::jit::more::mix;

#define REGISTER_MORE_KERNEL(key, func)                       \
  REGISTER_JITKERNEL_MORE(key, mix, mix::func##Kernel<float>, \
                          mix::func##Kernel<double>)

REGISTER_MORE_KERNEL(vsigmoid, VSigmoid);
REGISTER_MORE_KERNEL(vtanh, VTanh);

#undef REGISTER_MORE_KERNEL
