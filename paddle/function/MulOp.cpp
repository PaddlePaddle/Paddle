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
inline void vecAddTo(real* a, const real* b, size_t len) {
  for (unsigned int i = 0; i < len; ++i) {
    a[i] += b[i];
  }
}

inline void vecAddTo(real* a, const real* b, real scaleB, size_t len) {
  for (unsigned int i = 0; i < len; ++i) {
    a[i] += scaleB * b[i];
  }
}

inline void colVecAddTo(
    real* a, const real* b, size_t len, size_t aWidth, size_t bWidth) {
  for (unsigned int i = 0; i < len; ++i) {
    a[i * aWidth] += b[i * bWidth];
  }
}

inline void colVecAddTo(
    real* a, real* b, real c, size_t len, size_t aWidth, size_t bWidth) {
  for (unsigned int i = 0; i < len; ++i) {
    a[i * aWidth] += b[i * bWidth] * c;
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
  /// todo(tianbing), clean the code
  CHECK(!out.isTransposed()) << "Not supported";
  CHECK_EQ(out.getValueType(), FLOAT_VALUE);

  const real* A = a.getData();
  const real* B = b.getData();
  real* C = out.getValue();
  int* rows = out.getRows();
  int* cols = out.getCols();
  size_t height = out.getHeight();
  size_t width = out.getWidth();
  if (scaleT == 0) {
    out.zeroMem();
  }

  if (!a.isTransposed() && !b.isTransposed()) {
    size_t m = a.getWidth();
    CHECK_EQ(b.getHeight(), m);
    CHECK_EQ(a.getHeight(), height);
    CHECK_EQ(b.getWidth(), width);
    if (out.getFormat() == SPARSE_CSC) {
      for (size_t i = 0; i < width; i++) {
        size_t start = out.getColStartIdx(i);
        size_t end = out.getColStartIdx(i + 1);
        for (size_t j = start; j < end; j++) {
          real sum = 0;
          size_t rowIdx = rows[j];
          for (size_t k = 0; k < m; k++) {
            sum += A[rowIdx * m + k] * B[k * width + i];
          }
          C[j] = scaleAB * sum + scaleT * C[j];
        }
      }
    } else {
      for (size_t i = 0; i < height; i++) {
        size_t start = out.getRowStartIdx(i);
        size_t end = out.getRowStartIdx(i + 1);
        for (size_t j = start; j < end; j++) {
          real sum = 0;
          size_t colIdx = cols[j];
          for (size_t k = 0; k < m; k++) {
            sum += A[i * m + k] * B[k * width + colIdx];
          }
          C[j] = scaleAB * sum + scaleT * C[j];
        }
      }
    }
  } else if (a.isTransposed() && !b.isTransposed()) {
    size_t m = a.getHeight();
    CHECK_EQ(m, b.getHeight());
    CHECK_EQ(b.getWidth(), width);
    CHECK_EQ(a.getWidth(), height);

    if (out.getFormat() == SPARSE_CSC) {
      for (size_t i = 0; i < width; i++) {
        size_t start = out.getColStartIdx(i);
        size_t end = out.getColStartIdx(i + 1);
        for (size_t j = start; j < end; j++) {
          real sum = 0;
          size_t rowIdx = rows[j];
          for (size_t k = 0; k < m; k++) {
            sum += A[k * height + rowIdx] * B[k * width + i];
          }
          C[j] = scaleAB * sum + scaleT * C[j];
        }
      }
    } else {
      for (size_t i = 0; i < height; i++) {
        int start = out.getRowStartIdx(i);
        int end = out.getRowStartIdx(i + 1);
        for (int j = start; j < end; j++) {
          real sum = 0;
          size_t colIdx = cols[j];
          for (size_t k = 0; k < m; k++) {
            sum += A[k * height + i] * B[k * width + colIdx];
          }
          C[j] = scaleAB * sum + scaleT * C[j];
        }
      }
    }
  } else if (!a.isTransposed() && b.isTransposed()) {
    size_t m = a.getWidth();
    CHECK_EQ(b.getWidth(), m);
    CHECK_EQ(a.getHeight(), height);
    CHECK_EQ(b.getHeight(), width);
    if (out.getFormat() == SPARSE_CSR) {
      for (size_t i = 0; i < height; i++) {
        size_t start = out.getRowStartIdx(i);
        size_t end = out.getRowStartIdx(i + 1);
        for (size_t j = start; j < end; j++) {
          real sum = 0;
          size_t colIdx = cols[j];
          for (size_t k = 0; k < m; k++) {
            sum += A[i * m + k] * B[colIdx * m + k];
          }
          C[j] = scaleAB * sum + scaleT * C[j];
        }
      }
    } else {
      LOG(FATAL) << "Not supported csc format "
                    "when a is not trans and b is trans";
    }
  } else {
    LOG(FATAL) << "Not supported";
  }
}

template <>
void MulOp<DEVICE_TYPE_CPU>(CpuMatrix& out,
                            const CpuMatrix& a,
                            const CpuMatrix& b,
                            real scaleAB,
                            real scaleT) {
  /// todo(tianbing), clean the code
  CHECK(!out.isTransposed()) << "Not supported";
  CBLAS_TRANSPOSE aTrans = CblasNoTrans;
  size_t aRow = a.getHeight();
  size_t aCol = a.getWidth();
  CBLAS_TRANSPOSE bTrans = CblasNoTrans;
  size_t bRow = b.getHeight();
  size_t bCol = b.getWidth();
  if (a.isTransposed()) {
    aTrans = CblasTrans;
    aRow = a.getWidth();
    aCol = a.getHeight();
  }
  if (b.isTransposed()) {
    bTrans = CblasTrans;
    bRow = b.getWidth();
    bCol = b.getHeight();
  }

  /// C = A * B, for matrix format
  CHECK_EQ(aCol, bRow);
  CHECK_EQ(aRow, out.getHeight());
  CHECK_EQ(bCol, out.getWidth());

  const real* A = a.getData();
  const real* B = b.getData();
  real* C = out.getData();

  int M = out.getHeight();
  int N = out.getWidth();
  int K = aCol;
  int lda = a.getStride();
  int ldb = b.getStride();
  int ldc = out.getStride();

  GEMM(aTrans, bTrans, M, N, K, scaleAB, A, lda, B, ldb, scaleT, C, ldc);

  VLOG(2) << " A[0]=" << A[0] << " A[1]=" << A[1] << " B[0]=" << B[0]
          << " B[1]=" << B[1] << " C[0]=" << C[0] << " C[1]=" << C[1];
}

static ThreadLocal<std::vector<const real*>> threadLocalColArray;

template <>
void MulOp<DEVICE_TYPE_CPU>(CpuMatrix& out,
                            const CpuSparseMatrix& a,
                            const CpuMatrix& b,
                            real scaleAB,
                            real scaleT) {
  /// todo(tianbing), clean the code
  CHECK(!out.isTransposed()) << "Not supported";
  CHECK(!b.isTransposed()) << "Not supported";
  CHECK(scaleT == 0 || scaleT == 1) << "Not support";
  CHECK_EQ(scaleAB, static_cast<real>(1.0)) << "Not supported";
  CHECK_EQ(a.getFormat(), SPARSE_CSR) << "Not supported";

  const real* B = b.getData();
  real* C = out.getData();
  size_t height = out.getHeight();
  size_t width = out.getWidth();
  int* cols = a.getCols();
  real* values = a.getValue();

  if (scaleT == 0) {
    out.zeroMem();
  }

  if (!a.isTransposed()) {
    size_t m = a.getWidth();
    CHECK_EQ(b.getHeight(), m);
    CHECK_EQ(a.getHeight(), height);
    CHECK_EQ(b.getWidth(), width);

    if (a.getValueType() == NO_VALUE) {
      if (width % 32 == 0) {  // use libaddto
        CHECK_EQ((size_t)B % 32, 0UL);
        CHECK_EQ((size_t)C % 32, 0UL);
        auto& colArray = *threadLocalColArray;
        for (size_t i = 0; i < a.getHeight(); ++i) {
          const int start = a.getRowStartIdx(i);
          const int end = a.getRowStartIdx(i + 1);
          size_t colNum = end - start;
          colArray.resize(colNum);
          for (int j = 0; j < end - start; ++j) {
            colArray[j] = const_cast<CpuMatrix&>(b).getRow(cols[j + start]);
          }
          simd::batchAddTo(out.getRow(i), &colArray[0], colNum, width);
        }

      } else {
        for (size_t i = 0; i < a.getHeight(); ++i) {
          const int start = a.getRowStartIdx(i);
          const int end = a.getRowStartIdx(i + 1);
          for (int j = start; j < end; ++j) {
            vecAddTo(out.getRow(i),
                     const_cast<CpuMatrix&>(b).getRow(cols[j]),
                     width);
          }
        }
      }
    } else if (a.getValueType() == FLOAT_VALUE) {
      for (size_t i = 0; i < a.getHeight(); ++i) {
        const int start = a.getRowStartIdx(i);
        const int end = a.getRowStartIdx(i + 1);
        for (int j = start; j < end; ++j) {
          vecAddTo(out.getRow(i),
                   const_cast<CpuMatrix&>(b).getRow(cols[j]),
                   values[j],
                   width);
        }
      }
    }
  } else /*if (a->isTransposed())*/ {
    size_t m = a.getHeight();
    CHECK_EQ(b.getHeight(), m);
    CHECK_EQ(a.getWidth(), height);
    CHECK_EQ(b.getWidth(), width);
    if (a.getValueType() == NO_VALUE) {
      if (width % 32 == 0) {  // use libaddto
        CHECK_EQ((size_t)B % 32, 0UL);
        CHECK_EQ((size_t)C % 32, 0UL);
        for (size_t i = 0; i < a.getHeight(); ++i) {
          const int start = a.getRowStartIdx(i);
          const int end = a.getRowStartIdx(i + 1);
          for (int j = start; j < end; ++j) {
            simd::addTo(out.getRow(cols[j]),
                        const_cast<CpuMatrix&>(b).getRow(i),
                        width);
          }
        }

      } else {
        for (size_t i = 0; i < a.getHeight(); ++i) {
          const int start = a.getRowStartIdx(i);
          const int end = a.getRowStartIdx(i + 1);
          for (int j = start; j < end; ++j) {
            vecAddTo(out.getRow(cols[j]),
                     const_cast<CpuMatrix&>(b).getRow(i),
                     width);
          }
        }
      }
    } else if (a.getValueType() == FLOAT_VALUE) {
      for (size_t i = 0; i < a.getHeight(); ++i) {
        const int start = a.getRowStartIdx(i);
        const int end = a.getRowStartIdx(i + 1);
        for (int j = start; j < end; ++j) {
          vecAddTo(out.getRow(cols[j]),
                   const_cast<CpuMatrix&>(b).getRow(i),
                   values[j],
                   width);
        }
      }
    }
  }
}

template <>
void MulOp<DEVICE_TYPE_CPU>(CpuMatrix& out,
                            const CpuMatrix& a,
                            const CpuSparseMatrix& b,
                            real scaleAB,
                            real scaleT) {
  /// todo(tianbing), clean the code
  CHECK(!out.trans_) << "Not supported";
  CHECK(!a.isTransposed()) << "Not supported";
  CHECK(scaleT == 0 || scaleT == 1);
  CHECK_EQ(scaleAB, static_cast<real>(1.0));

  real* A = const_cast<real*>(a.getData());
  real* B = const_cast<real*>(b.getValue());
  real* C = out.getData();
  int* rows = b.getRows();
  int* cols = b.getCols();

  if (scaleT == 0) {
    out.zeroMem();
  }
  /// todo(tianbing), clean the code
  if (b.getFormat() == SPARSE_CSC) {
    if (!b.isTransposed()) {
      size_t m = a.getWidth();
      CHECK_EQ(b.getHeight(), m);
      CHECK_EQ(a.getHeight(), out.height_);
      CHECK_EQ(b.getWidth(), out.width_);

      if (b.getValueType() == NO_VALUE) {
        for (size_t j = 0; j < b.getWidth(); ++j) {
          int start = b.getColStartIdx(j);
          int end = b.getColStartIdx(j + 1);
          for (int i = start; i < end; ++i) {
            colVecAddTo(
                C + j, A + rows[i], out.height_, out.width_, a.getWidth());
          }
        }
      } else if (b.getValueType() == FLOAT_VALUE) {
        for (size_t j = 0; j < b.getWidth(); ++j) {
          int start = b.getColStartIdx(j);
          int end = b.getColStartIdx(j + 1);
          for (int i = start; i < end; ++i) {
            colVecAddTo(C + j,
                        A + rows[i],
                        B[i],
                        out.height_,
                        out.width_,
                        a.getWidth());
          }
        }
      }
    } else /*if (b.isTransposed())*/ {
      size_t m = a.getWidth();
      CHECK_EQ(b.getHeight(), out.width_);
      CHECK_EQ(a.getHeight(), out.height_);
      CHECK_EQ(b.getWidth(), m);
      if (b.getValueType() == NO_VALUE) {
        for (size_t i = 0; i < b.getWidth(); ++i) {
          int start = b.getColStartIdx(i);
          int end = b.getColStartIdx(i + 1);
          for (int j = start; j < end; ++j) {
            colVecAddTo(
                C + rows[j], A + i, out.height_, out.width_, a.getWidth());
          }
        }
      } else if (b.getValueType() == FLOAT_VALUE) {
        for (size_t i = 0; i < b.getWidth(); ++i) {
          int start = b.getColStartIdx(i);
          int end = b.getColStartIdx(i + 1);
          for (int j = start; j < end; ++j) {
            colVecAddTo(C + rows[j],
                        A + i,
                        B[j],
                        out.height_,
                        out.width_,
                        a.getWidth());
          }
        }
      }
    }
  } else {
    if (!b.isTransposed()) {
      size_t m = a.getWidth();
      CHECK_EQ(b.getHeight(), m);
      CHECK_EQ(a.getHeight(), out.height_);
      CHECK_EQ(b.getWidth(), out.width_);

      if (b.getValueType() == NO_VALUE) {
        for (size_t j = 0; j < b.getHeight(); ++j) {
          int start = b.getRowStartIdx(j);
          int end = b.getRowStartIdx(j + 1);
          for (int i = start; i < end; ++i) {
            colVecAddTo(
                C + cols[i], A + j, out.height_, out.width_, a.getWidth());
          }
        }
      } else if (b.getValueType() == FLOAT_VALUE) {
        for (size_t j = 0; j < b.getHeight(); ++j) {
          int start = b.getRowStartIdx(j);
          int end = b.getRowStartIdx(j + 1);
          for (int i = start; i < end; ++i) {
            colVecAddTo(C + cols[i],
                        A + j,
                        B[i],
                        out.height_,
                        out.width_,
                        a.getWidth());
          }
        }
      }
    } else /*if (b.isTransposed())*/ {
      size_t m = a.getWidth();
      CHECK_EQ(b.getHeight(), out.width_);
      CHECK_EQ(a.getHeight(), out.height_);
      CHECK_EQ(b.getWidth(), m);
      if (b.getValueType() == NO_VALUE) {
        for (size_t i = 0; i < b.getHeight(); ++i) {
          int start = b.getRowStartIdx(i);
          int end = b.getRowStartIdx(i + 1);
          for (int j = start; j < end; ++j) {
            colVecAddTo(
                C + i, A + cols[j], out.height_, out.width_, a.getWidth());
          }
        }
      } else if (b.getValueType() == FLOAT_VALUE) {
        for (size_t i = 0; i < b.getHeight(); ++i) {
          int start = b.getRowStartIdx(i);
          int end = b.getRowStartIdx(i + 1);
          for (int j = start; j < end; ++j) {
            colVecAddTo(C + i,
                        A + cols[j],
                        B[j],
                        out.height_,
                        out.width_,
                        a.getWidth());
          }
        }
      }
    }
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

    auto out_mat = outputs[0].matrix<Device>();
    /// matrix = matrix * matrix
    if (!inputs[0].isSparseArg() && !inputs[1].isSparseArg() &&
        !outputs[0].isSparseArg()) {
      MulOp<Device>(out_mat,
                    inputs[0].matrix<Device>(),
                    inputs[1].matrix<Device>(),
                    alpha_,
                    beta_);
      return;
    }

    /// matrix = matrix * sparse matrix
    if (!inputs[0].isSparseArg() && inputs[1].isSparseArg() &&
        !outputs[0].isSparseArg()) {
      MulOp<Device>(out_mat,
                    inputs[0].matrix<Device>(),
                    inputs[1].sparse().SparseMatrix<Device>(),
                    alpha_,
                    beta_);
      return;
    }

    /// matrix = sparse matrix * matrix
    if (inputs[0].isSparseArg() && !inputs[1].isSparseArg() &&
        !outputs[0].isSparseArg()) {
      MulOp<Device>(out_mat,
                    inputs[0].sparse().SparseMatrix<Device>(),
                    inputs[1].matrix<Device>(),
                    alpha_,
                    beta_);
      return;
    }

    /// sparse matrix = matrix * matrix
    auto out_sparse_mat = outputs[0].sparse().SparseMatrix<Device>();
    if (!inputs[0].isSparseArg() && !inputs[1].isSparseArg() &&
        outputs[0].isSparseArg()) {
      MulOp<Device>(out_sparse_mat,
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
