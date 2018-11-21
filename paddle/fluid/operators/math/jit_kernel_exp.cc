/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/jit_kernel.h"
#include <string>
#include "paddle/fluid/operators/math/jit_kernel_macro.h"
#include "paddle/fluid/operators/math/jit_kernel_refer.h"

#ifdef PADDLE_WITH_XBYAK
#include "paddle/fluid/operators/math/jit_code.h"
#endif

#ifdef PADDLE_WITH_MKLML
#include "paddle/fluid/platform/dynload/mklml.h"
#endif

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {
namespace jit = platform::jit;

#ifdef PADDLE_WITH_MKLML
// try to use MKL to speedup
template <typename T>
void VExpMKL(const T* x, T* y, int n);

template <>
void VExpMKL<float>(const float* x, float* y, int n) {
  platform::dynload::vsExp(n, x, y);
}

template <>
void VExpMKL<double>(const double* x, double* y, int n) {
  platform::dynload::vdExp(n, x, y);
}

template <typename T>
void VSigmoidMKL(const T* x, T* y, int n) {
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    y[i] = static_cast<T>(0) - y[i];
  }
  VExpMKL(y, y, n);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(1) / (static_cast<T>(1) + y[i]);
  }
}

template <typename T>
void VTanhMKL(const T* x, T* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(2) * x[i];
  }
  VSigmoidMKL(y, y, n);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(2) * y[i] - static_cast<T>(1);
  }
}
#endif

/* VExp JitKernel */
template <typename T>
class VExpKernelImpl : public VExpKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit VExpKernelImpl(int d) : VExpKernel<T>() {
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(d)) {
      size_t sz = 96 + d / YMM_FLOAT_BLOCK * 70 * 8;
      jitcode_.reset(new gen::VActJitCode(d, gen::operand_type::exp,
                                          sz > 4096 ? sz : 4096));
      this->Compute = jitcode_->getCode<void (*)(const T*, T*, int)>();
      return;
    }
#endif
#ifdef PADDLE_WITH_MKLML
    if (useMKL(d)) {
      this->Compute = VExpMKL<T>;
      return;
    }
#endif
    this->Compute = refer::VExp<T>;
  }

#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::VActJitCode> jitcode_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool VExpKernelImpl<float>::useJIT(int d) {
  return gen::VActJitCode::init(d, gen::operand_type::exp);
}
#endif

#ifdef PADDLE_WITH_MKLML
template <>
bool VExpKernelImpl<float>::useMKL(int d) {
  return d > 512;
}

template <>
bool VExpKernelImpl<double>::useMKL(int d) {
  return true;
}

#endif

/* VSigmoid JitKernel */
template <typename T>
class VSigmoidKernelImpl : public VSigmoidKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit VSigmoidKernelImpl(int d) : VSigmoidKernel<T>() {
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(d)) {
      size_t sz = 96 + d / YMM_FLOAT_BLOCK * 82 * 8;
      jitcode_.reset(new gen::VActJitCode(d, gen::operand_type::sigmoid,
                                          sz > 4096 ? sz : 4096));
      this->Compute = jitcode_->getCode<void (*)(const T*, T*, int)>();
      return;
    }
#endif

#ifdef PADDLE_WITH_MKLML
    // strictly it's a better impl with MKL, then is refer
    if (useMKL(d)) {
      this->Compute = VSigmoidMKL<T>;
      return;
    }
#endif
    this->Compute = refer::VSigmoid<T>;
  }

#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::VActJitCode> jitcode_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool VSigmoidKernelImpl<float>::useJIT(int d) {
  return gen::VActJitCode::init(d, gen::operand_type::sigmoid);
}
#endif

#ifdef PADDLE_WITH_MKLML
template <>
bool VSigmoidKernelImpl<float>::useMKL(int d) {
  return d > 512;
}

template <>
bool VSigmoidKernelImpl<double>::useMKL(int d) {
  return true;
}
#endif

/* VTanh JitKernel */
template <typename T>
class VTanhKernelImpl : public VTanhKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit VTanhKernelImpl(int d) : VTanhKernel<T>() {
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(d)) {
      size_t sz = 96 + d / YMM_FLOAT_BLOCK * 84 * 8;
      jitcode_.reset(new gen::VActJitCode(d, gen::operand_type::tanh,
                                          sz > 4096 ? sz : 4096));
      this->Compute = jitcode_->getCode<void (*)(const T*, T*, int)>();
      return;
    }
#endif

#ifdef PADDLE_WITH_MKLML
    // strictly it's a better impl with MKL, then is refer
    if (useMKL(d)) {
      this->Compute = VTanhMKL<T>;
      return;
    }
#endif
    this->Compute = refer::VTanh<T>;
  }

#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::VActJitCode> jitcode_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool VTanhKernelImpl<float>::useJIT(int d) {
  return gen::VActJitCode::init(d, gen::operand_type::tanh);
}
#endif

#ifdef PADDLE_WITH_MKLML
template <>
bool VTanhKernelImpl<float>::useMKL(int d) {
  return d > 512;
}

template <>
bool VTanhKernelImpl<double>::useMKL(int d) {
  return true;
}
#endif

REGISTER_JITKERNEL(vexp, VExpKernel);
REGISTER_JITKERNEL(vsigmoid, VSigmoidKernel);
REGISTER_JITKERNEL(vtanh, VTanhKernel);

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
