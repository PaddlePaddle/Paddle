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

#include "MulOp.h"
#include "GemmFunctor.h"
#include "paddle/math/SIMDFunctions.h"
#include "paddle/utils/ThreadLocal.h"

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
/// sparse matrix (+)= dense matrix * dense matrix
template <>
void MulOp<DEVICE_TYPE_CPU>(CpuSparseMatrix& out,
                            const CpuMatrix& a,
                            const CpuMatrix& b,
                            real scaleAB,
                            real scaleT,
                            bool aTrans,
                            bool bTrans) {
  CHECK_EQ(out.getValueType(), FLOAT_VALUE);
  if (scaleT == 0) {
    out.zeroMem();
  }
  const real* A = a.getData();
  const real* B = b.getData();
  real* C = out.getValue();
  int* rows = out.getRows();
  int* cols = out.getCols();
  size_t width = out.getWidth();
  size_t height = out.getHeight();

  /// SPARSE_CSC, {a any, b not trans}
  if (out.getFormat() == SPARSE_CSC) {
    /// b not trans and a any
    CHECK(!bTrans);
    size_t m = !aTrans ? a.getWidth() : a.getHeight();
    for (size_t i = 0; i < width; i++) {
      size_t start = out.getColStartIdx(i);
      size_t end = out.getColStartIdx(i + 1);
      for (size_t j = start; j < end; j++) {
        real sum = 0;
        size_t rowIdx = rows[j];
        for (size_t k = 0; k < m; k++) {
          sum += (!aTrans ? A[rowIdx * m + k] : A[k * height + rowIdx]) *
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
    CHECK(!(aTrans && bTrans));
    size_t m = a.getWidth();
    for (size_t i = 0; i < height; i++) {
      size_t start = out.getRowStartIdx(i);
      size_t end = out.getRowStartIdx(i + 1);
      for (size_t j = start; j < end; j++) {
        real sum = 0;
        size_t colIdx = cols[j];
        for (size_t k = 0; k < m; k++) {
          sum += (!aTrans ? A[i * m + k] : A[k * height + i]) *
                 (!bTrans ? B[k * width + colIdx] : B[colIdx * m + k]);
        }
        C[j] = scaleAB * sum + scaleT * C[j];
      }
    }
    return;
  }
}

/// dense matrix (+)= dense matrix * dense matrix
template <>
void MulOp<DEVICE_TYPE_CPU>(CpuMatrix& out,
                            const CpuMatrix& a,
                            const CpuMatrix& b,
                            real scaleAB,
                            real scaleT,
                            bool aTrans,
                            bool bTrans) {
  BlasGemm<DEVICE_TYPE_CPU, real>::compute(
      aTrans,
      bTrans,
      out.getHeight(),
      out.getWidth(),
      !aTrans ? a.getWidth() : a.getHeight(),
      scaleAB,
      a.getData(),
      a.getStride(),
      b.getData(),
      b.getStride(),
      scaleT,
      out.getData(),
      out.getStride());
}

/// dense matrix (+)= sparse matrix * dense matrix
template <>
void MulOp<DEVICE_TYPE_CPU>(CpuMatrix& out,
                            const CpuSparseMatrix& a,
                            const CpuMatrix& b,
                            real scaleAB,
                            real scaleT,
                            bool aTrans,
                            bool bTrans) {
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
      vecAddTo(!aTrans ? out.getRow(i) : out.getRow(cols[j]),
               !aTrans ? const_cast<CpuMatrix&>(b).getRow(cols[j])
                       : const_cast<CpuMatrix&>(b).getRow(i),
               (a.getValueType() == FLOAT_VALUE) ? values[j] : (real)1.0,
               out.getWidth());
    }
  }
}

/// dense matrix (+)= dense matrix * sparse matrix
template <>
void MulOp<DEVICE_TYPE_CPU>(CpuMatrix& out,
                            const CpuMatrix& a,
                            const CpuSparseMatrix& b,
                            real scaleAB,
                            real scaleT,
                            bool aTrans,
                            bool bTrans) {
  if (scaleT == 0) {
    out.zeroMem();
  }
  real* A = const_cast<real*>(a.getData());
  real* B = const_cast<real*>(b.getValue());
  real* C = out.getData();
  int* rows = b.getRows();
  int* cols = b.getCols();

  /// SPARSE_CSC format
  if (b.getFormat() == SPARSE_CSC) {
    for (size_t j = 0; j < b.getWidth(); ++j) {
      int start = b.getColStartIdx(j);
      int end = b.getColStartIdx(j + 1);
      for (int i = start; i < end; ++i) {
        colVecAddTo(!bTrans ? C + j : C + rows[i],
                    !bTrans ? A + rows[i] : A + j,
                    (b.getValueType() == NO_VALUE) ? (real)1.0 : B[i],
                    out.getHeight(),
                    out.getWidth(),
                    a.getWidth());
      }
    }
    return;
  }

  /// SPARSE_CSR format
  if (b.getFormat() == SPARSE_CSR) {
    for (size_t j = 0; j < b.getHeight(); ++j) {
      int start = b.getRowStartIdx(j);
      int end = b.getRowStartIdx(j + 1);
      for (int i = start; i < end; ++i) {
        colVecAddTo(!bTrans ? C + cols[i] : C + j,
                    !bTrans ? A + j : A + cols[i],
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
 * out = scaleT * out + scaleAB * (A * B)
 * here, scaleT in {0, 1}, scaleAB == 1,
 * out = A * B, ASSIGN_TO
 * out += A * B, ADD_TO
 *
 *
 * \param outputs[0]      output matrix (out), M * N,
 *                        could be either Sparse or Dense Matrix
 *                        M is num of rows, N is num of columns
 * \param inputs[0]       first input matrix (A),  M * K (if non-trans)
 *                        could be either Sparse or Dense Matrix
 *                        M is num of rows, K is num of columns
 * \param inputs[1]       second input matrix (B), K * N (if non-trans)
 *                        could be either Sparse or Dense Matrix
 *                        K is num of rows, N is num of columns
 *
 * Support eight Mul operators, with both GPU and CPU devices
 * For each device, four Mul operators are supported:
 * 1. dense (out) = dense (A) * dense (B)
 * 2. dense (out) = sparse (A) * dense (B)
 *    sparse matrix only support SPARSE_CSR format
 * 3. dense (out) = dense (A) * sparse (B)
 *    sparse matrix support SPARSE_CSC and SPARSE_CSR formats
 * 4. sparse (out) = dense (A) * dense (B)
 *    sparse matrix support SPARSE_CSC and SPARSE_CSR formats
 *
 */
template <DeviceType Device>
class MulFunc : public FunctionBase {
 public:
  void init(const FuncConfig& config) override {
    aTrans_ = config.get<bool>("aTrans");
    bTrans_ = config.get<bool>("bTrans");
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK(!aTrans_ || !bTrans_)
        << "Not support both a and b are transpose matrices";

    CHECK_EQ((size_t)2, inputs.size());
    CHECK_EQ((size_t)1, outputs.size());
    CHECK(inputs[0].data() && inputs[1].data() && outputs[0].data());
    CHECK_EQ(inputs[0].shape().ndims(), (size_t)2);
    CHECK_EQ(inputs[1].shape().ndims(), (size_t)2);
    CHECK_EQ(outputs[0].shape().ndims(), (size_t)2);

    size_t aRow = !aTrans_ ? inputs[0].shape()[0] : inputs[0].shape()[1];
    size_t aCol = !aTrans_ ? inputs[0].shape()[1] : inputs[0].shape()[0];
    size_t bRow = !bTrans_ ? inputs[1].shape()[0] : inputs[1].shape()[1];
    size_t bCol = !bTrans_ ? inputs[1].shape()[1] : inputs[1].shape()[0];
    /// C = A * B, or C += A * B, for matrix format
    CHECK_EQ(aCol, bRow);
    CHECK_EQ(aRow, outputs[0].shape()[0]);
    CHECK_EQ(bCol, outputs[0].shape()[1]);

    /// only support C = A * B (ASSIGN_TO) or C += A * B (ADD_TO)
    real scaleT = (outputs[0].getArgType() == ADD_TO) ? 1.0 : 0.0;

    /// support dense = not both sparse * sparse
    /// or sparse = dense * dense
    CHECK((!outputs[0].isSparseArg() &&
           !(inputs[0].isSparseArg() && inputs[1].isSparseArg())) ||
          (outputs[0].isSparseArg() && !inputs[0].isSparseArg() &&
           !inputs[1].isSparseArg()));

    auto outMat = outputs[0].matrix<Device>();
    /// dense matrix = dense matrix * dense matrix
    if (!inputs[0].isSparseArg() && !inputs[1].isSparseArg() &&
        !outputs[0].isSparseArg()) {
      MulOp<Device>(outMat,
                    inputs[0].matrix<Device>(),
                    inputs[1].matrix<Device>(),
                    1.0,  // scaleAB
                    scaleT,
                    aTrans_,
                    bTrans_);
      return;
    }

    /// dense matrix = dense matrix * sparse matrix
    if (!inputs[0].isSparseArg() && inputs[1].isSparseArg() &&
        !outputs[0].isSparseArg()) {
      CHECK(!aTrans_) << "Not supported a transpose";
      MulOp<Device>(outMat,
                    inputs[0].matrix<Device>(),
                    inputs[1].sparse().SparseMatrix<Device>(),
                    1.0,  // scaleAB
                    scaleT,
                    aTrans_,
                    bTrans_);
      return;
    }

    /// dense matrix = sparse matrix * dense matrix
    if (inputs[0].isSparseArg() && !inputs[1].isSparseArg() &&
        !outputs[0].isSparseArg()) {
      CHECK(!bTrans_) << "Not supported b transpose";
      CHECK_EQ(inputs[0].sparse().dataFormat(), T_SPARSE_CSR)
          << "Only supported SPARSE_CSR format for sparse matrix a";
      MulOp<Device>(outMat,
                    inputs[0].sparse().SparseMatrix<Device>(),
                    inputs[1].matrix<Device>(),
                    1.0,  // scaleAB
                    scaleT,
                    aTrans_,
                    bTrans_);
      return;
    }

    /// sparse matrix = dense matrix * dense matrix
    auto outSparseMat = outputs[0].sparse().SparseMatrix<Device>();
    if (!inputs[0].isSparseArg() && !inputs[1].isSparseArg() &&
        outputs[0].isSparseArg()) {
      MulOp<Device>(outSparseMat,
                    inputs[0].matrix<Device>(),
                    inputs[1].matrix<Device>(),
                    1.0,  // scaleAB
                    scaleT,
                    aTrans_,
                    bTrans_);
      return;
    }
  }

 private:
  bool aTrans_;
  bool bTrans_;
};

REGISTER_TYPED_FUNC(MulOp, CPU, MulFunc);
#ifdef PADDLE_WITH_CUDA
REGISTER_TYPED_FUNC(MulOp, GPU, MulFunc);
#endif
}  // namespace paddle
