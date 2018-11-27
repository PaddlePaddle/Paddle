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
#include "paddle/fluid/platform/enforce.h"

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
template <typename T>
void VMulMKL(const T* x, const T* y, T* z, int n);

template <>
void VMulMKL<float>(const float* x, const float* y, float* z, int n) {
  platform::dynload::vsMul(n, x, y, z);
}

template <>
void VMulMKL<double>(const double* x, const double* y, double* z, int n) {
  platform::dynload::vdMul(n, x, y, z);
}

template <typename T>
void VAddMKL(const T* x, const T* y, T* z, int n);

template <>
void VAddMKL<float>(const float* x, const float* y, float* z, int n) {
  platform::dynload::vsAdd(n, x, y, z);
}

template <>
void VAddMKL<double>(const double* x, const double* y, double* z, int n) {
  platform::dynload::vdAdd(n, x, y, z);
}

template <typename T>
void VScalMKL(const T* a, const T* x, T* y, int n);

template <>
void VScalMKL<float>(const float* a, const float* x, float* y, int n) {
  if (x == y) {
    platform::dynload::cblas_sscal(n, *a, y, 1);
  } else {
    refer::VScal<float>(a, x, y, n);
  }
}

template <>
void VScalMKL<double>(const double* a, const double* x, double* y, int n) {
  if (x == y) {
    platform::dynload::cblas_dscal(n, *a, y, 1);
  } else {
    refer::VScal<double>(a, x, y, n);
  }
}

#endif

/* VMUL JitKernel */
template <typename T>
class VMulKernelImpl : public VMulKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit VMulKernelImpl(int d) : VMulKernel<T>() {
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(d)) {
      // roughly estimate the size of code
      size_t sz = 96 + d / YMM_FLOAT_BLOCK * 4 * 8;
      jitcode_.reset(new gen::VXXJitCode(d, gen::operand_type::mul, 0, false,
                                         sz > 4096 ? sz : 4096));
      this->Compute =
          jitcode_->getCode<void (*)(const T*, const T*, T*, int)>();
      return;
    }
#endif
#ifdef PADDLE_WITH_MKLML
    if (useMKL(d)) {
      this->Compute = VMulMKL<T>;
      return;
    }
#endif
    this->Compute = refer::VMul<T>;
  }

#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::VXXJitCode> jitcode_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool VMulKernelImpl<float>::useJIT(int d) {
  return gen::VXXJitCode::init(d);
}
#endif

#ifdef PADDLE_WITH_MKLML
template <>
bool VMulKernelImpl<float>::useMKL(int d) {
  return jit::MayIUse(jit::avx512f) && d > 512;
}

template <>
bool VMulKernelImpl<double>::useMKL(int d) {
  return true;
}
#endif

/* VAdd JitKernel */
template <typename T>
class VAddKernelImpl : public VAddKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit VAddKernelImpl(int d) : VAddKernel<T>() {
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(d)) {
      size_t sz = 96 + d / YMM_FLOAT_BLOCK * 4 * 8;
      jitcode_.reset(new gen::VXXJitCode(d, gen::operand_type::add, 0, false,
                                         sz > 4096 ? sz : 4096));
      this->Compute =
          jitcode_->getCode<void (*)(const T*, const T*, T*, int)>();
      return;
    }
#endif
#ifdef PADDLE_WITH_MKLML
    if (useMKL(d)) {
      this->Compute = VAddMKL<T>;
      return;
    }
#endif
    this->Compute = refer::VAdd<T>;
  }
#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::VXXJitCode> jitcode_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool VAddKernelImpl<float>::useJIT(int d) {
  return gen::VXXJitCode::init(d);
}
#endif

#ifdef PADDLE_WITH_MKLML
template <>
bool VAddKernelImpl<float>::useMKL(int d) {
  return d > 512;
}

template <>
bool VAddKernelImpl<double>::useMKL(int d) {
  return true;
}
#endif

#ifdef PADDLE_WITH_MKLDNN
/* EltwiseMul for nChw16c & NC inputs JitKernel */
template <typename T>
class EltwiseMulnChw16cNCKernelImpl
    : public math::jitkernel::EltwiseMulnChw16cNCKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit EltwiseMulnChw16cNCKernelImpl(int d)
      : EltwiseMulnChw16cNCKernel<T>() {
    using mul_func_t = void (*)(const float*, const float*, float*, int, int);
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(d)) {
      // roughly estimate the size of code
      size_t sz = 96 + d / YMM_FLOAT_BLOCK * 4 * 8;
      sz = sz > 4096 ? sz : 4096;
      jitcode_.reset(new gen::EltwiseMulnChw16cNC(sz));
      this->Compute = (mul_func_t)jitcode_->getCode();
      return;
    }
#endif
    PADDLE_THROW(
        "This kernel shouldn't be used in Non-Xbyak, Non-MKL-DNN "
        "environemnt");
  }

#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::EltwiseMulnChw16cNC> jitcode_{nullptr};
};

template <>
bool EltwiseMulnChw16cNCKernelImpl<float>::useJIT(int d) {
  return true;
}
#endif
#endif

/* VAddRelu JitKernel */
template <typename T>
class VAddReluKernelImpl : public VAddReluKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit VAddReluKernelImpl(int d) : VAddReluKernel<T>() {
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(d)) {
      size_t sz = 96 + d / YMM_FLOAT_BLOCK * 4 * 8;
      jitcode_.reset(new gen::VXXJitCode(d, gen::operand_type::add, 0, true,
                                         sz > 4096 ? sz : 4096));
      this->Compute =
          jitcode_->getCode<void (*)(const T*, const T*, T*, int)>();
      return;
    }
#endif
    this->Compute = refer::VAddRelu<T>;
  }
#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::VXXJitCode> jitcode_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool VAddReluKernelImpl<float>::useJIT(int d) {
  return gen::VXXJitCode::init(d);
}
#endif

/* VScal JitKernel */
template <typename T>
class VScalKernelImpl : public VScalKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit VScalKernelImpl(int d) : VScalKernel<T>() {
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(d)) {
      size_t sz = 96 + d / YMM_FLOAT_BLOCK * 4 * 8;
      jitcode_.reset(new gen::VXXJitCode(d, gen::operand_type::mul, 1, false,
                                         sz > 4096 ? sz : 4096));
      this->Compute =
          jitcode_->getCode<void (*)(const T*, const T*, T*, int)>();
      return;
    }
#endif
#ifdef PADDLE_WITH_MKLML
    if (useMKL(d)) {
      this->Compute = VScalMKL<T>;
      return;
    }
#endif
    this->Compute = refer::VScal<T>;
  }
#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::VXXJitCode> jitcode_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool VScalKernelImpl<float>::useJIT(int d) {
  return gen::VXXJitCode::init(d, 1);
}
#endif

#ifdef PADDLE_WITH_MKLML
template <>
bool VScalKernelImpl<float>::useMKL(int d) {
  return d > 512;
}
template <>
bool VScalKernelImpl<double>::useMKL(int d) {
  return true;
}
#endif

/* VAddBias JitKernel */
template <typename T>
class VAddBiasKernelImpl : public VAddBiasKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit VAddBiasKernelImpl(int d) : VAddBiasKernel<T>() {
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(d)) {
      size_t sz = 96 + d / YMM_FLOAT_BLOCK * 4 * 8;
      jitcode_.reset(new gen::VXXJitCode(d, gen::operand_type::add, 1, false,
                                         sz > 4096 ? sz : 4096));
      this->Compute =
          jitcode_->getCode<void (*)(const T*, const T*, T*, int)>();
      return;
    }
#endif

    this->Compute = refer::VAddBias<T>;
  }
#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::VXXJitCode> jitcode_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool VAddBiasKernelImpl<float>::useJIT(int d) {
  return gen::VXXJitCode::init(d, 1);
}
#endif

/* VRelu JitKernel */
template <typename T>
class VReluKernelImpl : public VReluKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit VReluKernelImpl(int d) : VReluKernel<T>() {
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(d)) {
      size_t sz = 96 /* init size */ +
                  d / YMM_FLOAT_BLOCK * 4 /* instructions */ *
                      8 /* average bytes for each instruction */;
      jitcode_.reset(new gen::VActJitCode(d, gen::operand_type::relu,
                                          sz > 4096 ? sz : 4096));
      this->Compute = jitcode_->getCode<void (*)(const T*, T*, int)>();
      return;
    }
#endif

    this->Compute = refer::VRelu<T>;
  }
#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::VActJitCode> jitcode_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool VReluKernelImpl<float>::useJIT(int d) {
  return gen::VActJitCode::init(d, gen::operand_type::relu);
}
#endif

/* An empty JitKernel */
template <typename T>
class VIdentityKernelImpl : public VIdentityKernel<T> {
 public:
  JITKERNEL_DECLARE_STATIC_FUNC;
  explicit VIdentityKernelImpl(int d) : VIdentityKernel<T>() {
    this->Compute = refer::VIdentity<T>;
  }
};

REGISTER_JITKERNEL(vmul, VMulKernel);
REGISTER_JITKERNEL(vadd, VAddKernel);
REGISTER_JITKERNEL(vaddrelu, VAddReluKernel);
REGISTER_JITKERNEL(vscal, VScalKernel);
REGISTER_JITKERNEL(vaddbias, VAddBiasKernel);
REGISTER_JITKERNEL(vrelu, VReluKernel);
REGISTER_JITKERNEL(videntity, VIdentityKernel);
#ifdef PADDLE_WITH_MKLDNN
REGISTER_JITKERNEL(eltwise_mul_nchw16c, EltwiseMulnChw16cNCKernel);
#endif

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
