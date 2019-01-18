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

#include "paddle/fluid/operators/jit/more/mkl/mkl.h"
#include "paddle/fluid/operators/jit/refer/refer.h"
#include "paddle/fluid/operators/jit/registry.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/dynload/mklml.h"

namespace paddle {
namespace operators {
namespace jit {
namespace more {
namespace mkl {

template <>
void MatMul<float>(const float* a, const float* b, float* c, int m, int n,
                   int k) {
  platform::dynload::cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m,
                                 n, k, 1.f, a, k, b, n, 0.f, c, n);
}

template <>
void MatMul<double>(const double* a, const double* b, double* c, int m, int n,
                    int k) {
  platform::dynload::cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m,
                                 n, k, 1.0, a, k, b, n, 0.0, c, n);
}

template <>
void VMul<float>(const float* x, const float* y, float* z, int n) {
  platform::dynload::vsMul(n, x, y, z);
}

template <>
void VMul<double>(const double* x, const double* y, double* z, int n) {
  platform::dynload::vdMul(n, x, y, z);
}

template <>
void VAdd<float>(const float* x, const float* y, float* z, int n) {
  platform::dynload::vsAdd(n, x, y, z);
}

template <>
void VAdd<double>(const double* x, const double* y, double* z, int n) {
  platform::dynload::vdAdd(n, x, y, z);
}

template <>
void VScal<float>(const float* a, const float* x, float* y, int n) {
  if (x == y) {
    platform::dynload::cblas_sscal(n, *a, y, 1);
  } else {
    refer::VScal<float>(a, x, y, n);
  }
}

template <>
void VScal<double>(const double* a, const double* x, double* y, int n) {
  if (x == y) {
    platform::dynload::cblas_dscal(n, *a, y, 1);
  } else {
    refer::VScal<double>(a, x, y, n);
  }
}

template <>
void VExp<float>(const float* x, float* y, int n) {
  platform::dynload::vsExp(n, x, y);
}

template <>
void VExp<double>(const double* x, double* y, int n) {
  platform::dynload::vdExp(n, x, y);
}

template <>
void VSquare<float>(const float* x, float* y, int n) {
  platform::dynload::vsSqr(n, x, y);
}

template <>
void VSquare<double>(const double* x, double* y, int n) {
  platform::dynload::vdSqr(n, x, y);
}

template <>
void VCopy<float>(const float* x, float* y, int n) {
  platform::dynload::cblas_scopy(n, x, 1, y, 1);
}

template <>
void VCopy<double>(const double* x, double* y, int n) {
  platform::dynload::cblas_dcopy(n, x, 1, y, 1);
}

template <>
void VAXPY<float>(float a, const float* x, float* y, int n) {
  platform::dynload::cblas_saxpy(n, a, x, 1, y, 1);
}

template <>
void VAXPY<double>(double a, const double* x, double* y, int n) {
  platform::dynload::cblas_daxpy(n, a, x, 1, y, 1);
}

// TODO(TJ): tuning me carefully on AVX, AVX2 and AVX512
template <>
bool MatMulKernel<float>::UseMe(const int& d) const {
  return platform::MayIUse(platform::avx);
}

template <>
bool VMulKernel<float>::UseMe(const int& d) const {
  return platform::MayIUse(platform::avx512f) && d > 512;
}

template <>
bool VAddKernel<float>::UseMe(const int& d) const {
  return platform::MayIUse(platform::avx512f) && d > 512;
}

template <>
bool VScalKernel<float>::UseMe(const int& d) const {
  return platform::MayIUse(platform::avx512f) && d > 512;
}

template <>
bool VExpKernel<float>::UseMe(const int& d) const {
  return d > 7;
}

template <>
bool VSquareKernel<float>::UseMe(const int& d) const {
  return d > 7;
}

template <>
bool VSigmoidKernel<float>::UseMe(const int& d) const {
  return d > 7;
}

template <>
bool VTanhKernel<float>::UseMe(const int& d) const {
  return d > 7;
}

template <>
bool SeqPoolKernel<float>::UseMe(const seq_pool_attr_t& attr) const {
  return true;
}

template <>
bool SeqPoolKernel<double>::UseMe(const seq_pool_attr_t& attr) const {
  return true;
}

#define AWALYS_USE_ME_WITH_DOUBLE(func)                  \
  template <>                                            \
  bool func##Kernel<double>::UseMe(const int& d) const { \
    return true;                                         \
  }

AWALYS_USE_ME_WITH_DOUBLE(MatMul);
AWALYS_USE_ME_WITH_DOUBLE(VMul);
AWALYS_USE_ME_WITH_DOUBLE(VAdd);
AWALYS_USE_ME_WITH_DOUBLE(VScal);
AWALYS_USE_ME_WITH_DOUBLE(VExp);
AWALYS_USE_ME_WITH_DOUBLE(VSigmoid);
AWALYS_USE_ME_WITH_DOUBLE(VTanh);
AWALYS_USE_ME_WITH_DOUBLE(VSquare);

#undef AWALYS_USE_ME_WITH_DOUBLE
}  // namespace mkl
}  // namespace more
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace mkl = paddle::operators::jit::more::mkl;

#define REGISTER_MKL_KERNEL(key, func)                        \
  REGISTER_JITKERNEL_MORE(key, mkl, mkl::func##Kernel<float>, \
                          mkl::func##Kernel<double>)

REGISTER_MKL_KERNEL(kMatMul, MatMul);
REGISTER_MKL_KERNEL(kVMul, VMul);
REGISTER_MKL_KERNEL(kVAdd, VAdd);
REGISTER_MKL_KERNEL(kVScal, VScal);
REGISTER_MKL_KERNEL(kVExp, VExp);
REGISTER_MKL_KERNEL(kVSquare, VSquare);
REGISTER_MKL_KERNEL(kVSigmoid, VSigmoid);
REGISTER_MKL_KERNEL(kVTanh, VTanh);
REGISTER_MKL_KERNEL(kSeqPool, SeqPool);

#undef REGISTER_MKL_KERNEL
