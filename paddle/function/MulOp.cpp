/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "MulOp.h"
/// todo(tianbing), delete it
#include <iostream>
#include "paddle/math/MathFunctions.h"
#include "paddle/math/SIMDFunctions.h"
#include "paddle/utils/ThreadLocal.h"

#ifndef PADDLE_TYPE_DOUBLE
#define GEMM paddle::gemm<float>
#else
#define GEMM paddle::gemm<double>
#endif

namespace {
inline void vecAddTo(real* a, const real* b, real scaleB, size_t len) {
  for (unsigned int i = 0; i < len; ++i) {
    a[i] += (1.0 == scaleB) ? b[i] : scaleB * b[i];
  }
}

inline void colVecAddTo(
    real* a, real* b, real c, size_t len, size_t aWidth, size_t bWidth) {
  for (unsigned int i = 0; i < len; ++i) {
    a[i * aWidth] += (1.0 == c) ? b[i * bWidth] : b[i * bWidth] * c;
  }
}
}  // namespace

namespace paddle {
template <>
void MulOp<DEVICE_TYPE_CPU>(CpuSparseMatrix& out,
                            const CpuMatrix& a,
                            const CpuMatrix& b,
                            real scaleAB,
                            real scaleT) {
  CHECK(!out.isTransposed()) << "Not supported";
  CHECK_EQ(out.getValueType(), FLOAT_VALUE);
  CHECK(!a.isTransposed() || !b.isTransposed())
      << "Not support both a and b are transpose matrices";

  size_t height = out.getHeight();
  size_t width = out.getWidth();
  size_t aRow = !a.isTransposed() ? a.getHeight() : a.getWidth();
  size_t aCol = !a.isTransposed() ? a.getWidth() : a.getHeight();
  size_t bRow = !b.isTransposed() ? b.getHeight() : b.getWidth();
  size_t bCol = !b.isTransposed() ? b.getWidth() : b.getHeight();
  /// C = A * B, for matrix format
  CHECK(aCol == bRow && aRow == height && bCol == width);

  if (scaleT == 0) {
    out.zeroMem();
  }
  const real* A = a.getData();
  const real* B = b.getData();
  real* C = out.getValue();
  int* rows = out.getRows();
  int* cols = out.getCols();

  /// SPARSE_CSC, {a any, b not trans}
  if (out.getFormat() == SPARSE_CSC) {
    /// b not trans and a any
    CHECK(!b.isTransposed());
    size_t m = !a.isTransposed() ? a.getWidth() : a.getHeight();
    for (size_t i = 0; i < width; i++) {
      size_t start = out.getColStartIdx(i);
      size_t end = out.getColStartIdx(i + 1);
      for (size_t j = start; j < end; j++) {
        real sum = 0;
        size_t rowIdx = rows[j];
        for (size_t k = 0; k < m; k++) {
          sum +=
              (!a.isTransposed() ? A[rowIdx * m + k] : A[k * height + rowIdx]) *
              B[k * width + i];
        }
        C[j] = scaleAB * sum + scaleT * C[j];
      }
    }
    return;
  }

  /// SPARSE_CSR, {a any, b not trans} or {a not trans, b trans}
  if (out.getFormat() == SPARSE_CSR) {
    /// a and b can not both transpose
    CHECK(!(a.isTransposed() && b.isTransposed()));
    size_t m = a.getWidth();
    for (size_t i = 0; i < height; i++) {
      size_t start = out.getRowStartIdx(i);
      size_t end = out.getRowStartIdx(i + 1);
      for (size_t j = start; j < end; j++) {
        real sum = 0;
        size_t colIdx = cols[j];
        for (size_t k = 0; k < m; k++) {
          sum +=
              (!a.isTransposed() ? A[i * m + k] : A[k * height + i]) *
              (!b.isTransposed() ? B[k * width + colIdx] : B[colIdx * m + k]);
        }
        C[j] = scaleAB * sum + scaleT * C[j];
      }
    }
    return;
  }
}

template <>
void MulOp<DEVICE_TYPE_CPU>(CpuMatrix& out,
                            const CpuMatrix& a,
                            const CpuMatrix& b,
                            real scaleAB,
                            real scaleT) {
  CHECK(!out.isTransposed()) << "out matrix transpose not supported";
  CBLAS_TRANSPOSE aTrans = a.isTransposed() ? CblasTrans : CblasNoTrans;
  size_t aRow = a.isTransposed() ? a.getWidth() : a.getHeight();
  size_t aCol = a.isTransposed() ? a.getHeight() : a.getWidth();
  CBLAS_TRANSPOSE bTrans = b.isTransposed() ? CblasTrans : CblasNoTrans;
  size_t bRow = b.isTransposed() ? b.getWidth() : b.getHeight();
  size_t bCol = b.isTransposed() ? b.getHeight() : b.getWidth();

  /// C = A * B, for matrix format
  CHECK_EQ(aCol, bRow);
  CHECK_EQ(aRow, out.getHeight());
  CHECK_EQ(bCol, out.getWidth());

  GEMM(aTrans,
       bTrans,
       out.getHeight(),
       out.getWidth(),
       aCol,
       scaleAB,
       a.getData(),
       a.getStride(),
       b.getData(),
       b.getStride(),
       scaleT,
       out.getData(),
       out.getStride());
}

template <>
void MulOp<DEVICE_TYPE_CPU>(CpuMatrix& out,
                            const CpuSparseMatrix& a,
                            const CpuMatrix& b,
                            real scaleAB,
                            real scaleT) {
  CHECK(!out.isTransposed()) << "Not supported";
  CHECK(!b.isTransposed()) << "Not supported";
  CHECK(scaleT == 0 || scaleT == 1) << "Not support";
  CHECK_EQ(scaleAB, static_cast<real>(1.0)) << "Not supported";
  CHECK_EQ(a.getFormat(), SPARSE_CSR) << "Not supported";

  if (!a.isTransposed()) {
    CHECK(b.getHeight() == a.getWidth() && a.getHeight() == out.getHeight() &&
          b.getWidth() == out.getWidth());
  } else {
    CHECK(b.getHeight() == a.getHeight() && a.getWidth() == out.getHeight() &&
          b.getWidth() == out.getWidth());
  }

  if (scaleT == 0) {
    out.zeroMem();
  }
  const real* B = b.getData();
  real* C = out.getData();
  if (out.getWidth() % 32 == 0) {
    CHECK_EQ((size_t)B % 32, 0UL);
    CHECK_EQ((size_t)C % 32, 0UL);
  }

  int* cols = a.getCols();
  real* values = a.getValue();
  for (size_t i = 0; i < a.getHeight(); ++i) {
    const int start = a.getRowStartIdx(i);
    const int end = a.getRowStartIdx(i + 1);
    for (int j = start; j < end; ++j) {
      vecAddTo(!a.isTransposed() ? out.getRow(i) : out.getRow(cols[j]),
               !a.isTransposed() ? const_cast<CpuMatrix&>(b).getRow(cols[j])
                                 : const_cast<CpuMatrix&>(b).getRow(i),
               (a.getValueType() == FLOAT_VALUE) ? values[j] : (real)1.0,
               out.getWidth());
    }
  }
}

template <>
void MulOp<DEVICE_TYPE_CPU>(CpuMatrix& out,
                            const CpuMatrix& a,
                            const CpuSparseMatrix& b,
                            real scaleAB,
                            real scaleT) {
  CHECK(!out.trans_) << "Not supported";
  CHECK(!a.isTransposed()) << "Not supported";
  CHECK(scaleT == 0 || scaleT == 1);
  CHECK_EQ(scaleAB, static_cast<real>(1.0));
  if (!b.isTransposed()) {  /// b is not Transpose
    CHECK(b.getHeight() == a.getWidth() && a.getHeight() == out.getHeight() &&
          b.getWidth() == out.getWidth());
  } else {
    CHECK(b.getHeight() == out.getWidth() && a.getHeight() == out.getHeight() &&
          b.getWidth() == a.getWidth());
  }

  if (scaleT == 0) {
    out.zeroMem();
  }
  real* A = const_cast<real*>(a.getData());
  real* B = const_cast<real*>(b.getValue());
  real* C = out.getData();
  int* rows = b.getRows();
  int* cols = b.getCols();

  /// b.getFormat() == SPARSE_CSC
  if (b.getFormat() == SPARSE_CSC) {
    for (size_t j = 0; j < b.getWidth(); ++j) {
      int start = b.getColStartIdx(j);
      int end = b.getColStartIdx(j + 1);
      for (int i = start; i < end; ++i) {
        colVecAddTo(!b.isTransposed() ? C + j : C + rows[i],
                    !b.isTransposed() ? A + rows[i] : A + j,
                    (b.getValueType() == NO_VALUE) ? (real)1.0 : B[i],
                    out.getHeight(),
                    out.getWidth(),
                    a.getWidth());
      }
    }
    return;
  }

  /// b.getFormat() == SPARSE_CSR
  if (b.getFormat() == SPARSE_CSR) {
    for (size_t j = 0; j < b.getHeight(); ++j) {
      int start = b.getRowStartIdx(j);
      int end = b.getRowStartIdx(j + 1);
      for (int i = start; i < end; ++i) {
        colVecAddTo(!b.isTransposed() ? C + cols[i] : C + j,
                    !b.isTransposed() ? A + j : A + cols[i],
                    (b.getValueType() == NO_VALUE) ? (real)1.0 : B[i],
                    out.getHeight(),
                    out.getWidth(),
                    a.getWidth());
      }
    }
    return;
  }
}

/**
 * mul operator
 * out = scaleT * out + scaleAB*(in1 * in2)
 *
 * \param outputs[0]      output matrix, M * N
 * \param inputs[0]       first input (sparse) matrix,  M * K (if non-trans)
 * \param inputs[1]       second input matrix, K * N (if non-trans)
 */
template <DeviceType Device>
class MulFunc : public FunctionBase {
public:
  void init(const FuncConfig& config) override {
    alpha_ = config.get<real>("scaleAB");
    beta_ = config.get<real>("scaleT");
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ((size_t)2, inputs.size());
    CHECK_EQ((size_t)1, outputs.size());
    CHECK(inputs[0].data() && inputs[1].data() && outputs[0].data());
    CHECK_EQ(inputs[0].shape().ndims(), (size_t)2);
    CHECK_EQ(inputs[1].shape().ndims(), (size_t)2);
    CHECK_EQ(outputs[0].shape().ndims(), (size_t)2);
    CHECK_EQ(outputs[0].getArgType(), ADD_TO);

    auto outMat = outputs[0].matrix<Device>();
    /// matrix = matrix * matrix
    if (!inputs[0].isSparseArg() && !inputs[1].isSparseArg() &&
        !outputs[0].isSparseArg()) {
      MulOp<Device>(outMat,
                    inputs[0].matrix<Device>(),
                    inputs[1].matrix<Device>(),
                    alpha_,
                    beta_);
      return;
    }

    /// matrix = matrix * sparse matrix
    if (!inputs[0].isSparseArg() && inputs[1].isSparseArg() &&
        !outputs[0].isSparseArg()) {
      MulOp<Device>(outMat,
                    inputs[0].matrix<Device>(),
                    inputs[1].sparse().SparseMatrix<Device>(),
                    alpha_,
                    beta_);
      return;
    }

    /// matrix = sparse matrix * matrix
    if (inputs[0].isSparseArg() && !inputs[1].isSparseArg() &&
        !outputs[0].isSparseArg()) {
      MulOp<Device>(outMat,
                    inputs[0].sparse().SparseMatrix<Device>(),
                    inputs[1].matrix<Device>(),
                    alpha_,
                    beta_);
      return;
    }

    /// sparse matrix = matrix * matrix
    auto outSparseMat = outputs[0].sparse().SparseMatrix<Device>();
    if (!inputs[0].isSparseArg() && !inputs[1].isSparseArg() &&
        outputs[0].isSparseArg()) {
      /*
      LOG(INFO) << "input0";
      inputs[0].matrix<Device>().print(std::cout);
      LOG(INFO) << "input1";
      inputs[1].matrix<Device>().print(std::cout);
      LOG(INFO) << "output sparse matrix";
      outSparseMat.print(std::cout); */
      MulOp<Device>(outSparseMat,
                    inputs[0].matrix<Device>(),
                    inputs[1].matrix<Device>(),
                    alpha_,
                    beta_);
      return;
    }
  }

private:
  real alpha_;
  real beta_;
};

REGISTER_TYPED_FUNC(MulOp, CPU, MulFunc);
#ifndef PADDLE_ONLY_CPU
REGISTER_TYPED_FUNC(MulOp, GPU, MulFunc);
#endif
}  // namespace paddle
