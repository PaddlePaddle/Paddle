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
#include "Register.h"
#include "paddle/math/MathFunctions.h"
#include "paddle/math/SIMDFunctions.h"
#include "paddle/topology/Attribute.h"
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
  GEMM(aTrans ? CblasTrans : CblasNoTrans,
       bTrans ? CblasTrans : CblasNoTrans,
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

struct MulAttribute : public topology::Attribute {
  bool aTrans;
  bool bTrans;

  REGISTER_FUNC_ATTRIBUTE() {
    regAttr(&MulAttribute::aTrans, "aTrans", "Matrix A is transposed or not");
    regAttr(&MulAttribute::bTrans, "bTrans", "Matrix B is transposed or not");
  }
};

template <DeviceType Device>
static Error mul(const BufferArgs& inputs,
                 const BufferArgs& outputs,
                 const MulAttribute& attr) {
  real scaleT = (outputs[0].getArgType() == ADD_TO) ? 1.0 : 0.0;
  auto outMat = outputs[0].matrix<Device>();
  /// dense matrix = dense matrix * dense matrix
  if (!inputs[0].isSparseArg() && !inputs[1].isSparseArg() &&
      !outputs[0].isSparseArg()) {
    MulOp<Device>(outMat,
                  inputs[0].matrix<Device>(),
                  inputs[1].matrix<Device>(),
                  1.0,  // scaleAB
                  scaleT,
                  attr.aTrans,
                  attr.bTrans);
  }

  /// dense matrix = dense matrix * sparse matrix
  if (!inputs[0].isSparseArg() && inputs[1].isSparseArg() &&
      !outputs[0].isSparseArg()) {
    MulOp<Device>(outMat,
                  inputs[0].matrix<Device>(),
                  inputs[1].sparse().SparseMatrix<Device>(),
                  1.0,  // scaleAB
                  scaleT,
                  attr.aTrans,
                  attr.bTrans);
  }

  /// dense matrix = sparse matrix * dense matrix
  if (inputs[0].isSparseArg() && !inputs[1].isSparseArg() &&
      !outputs[0].isSparseArg()) {
    MulOp<Device>(outMat,
                  inputs[0].sparse().SparseMatrix<Device>(),
                  inputs[1].matrix<Device>(),
                  1.0,  // scaleAB
                  scaleT,
                  attr.aTrans,
                  attr.bTrans);
  }
  return Error();
}

static Error MatrixShape(const std::vector<size_t>& a,
                         const std::vector<size_t>& b,
                         const MulAttribute& attr,
                         topology::Tensor& out) {
  size_t aRow, aCol, bRow, bCol;
  if (attr.aTrans) {
    aRow = a[1];
    aCol = a[0];
  } else {
    aRow = a[0];
    aCol = a[1];
  }
  if (attr.bTrans) {
    bRow = b[1];
    bCol = b[0];
  } else {
    bRow = b[0];
    bCol = b[1];
  }

  if (aCol != bRow) {
    return Error(
        "Matrix shape mismatch [%d, %d]x[%d, %d]", aRow, aCol, bRow, bCol);
  }
  out.setShape({aRow, bCol});
  return Error();
}

BEGIN_REGISTER_FUNCTION(Mul, mul, MulAttribute)
addTensor<INPUT>(2,
                 -1,
                 {topology::DataType::DENSE,
                  topology::DataType::SPARSE_INTEGER,
                  topology::DataType::SPARSE});
addTensor<INPUT>(2,
                 -1,
                 {topology::DataType::DENSE,
                  topology::DataType::SPARSE_INTEGER,
                  topology::DataType::SPARSE});
addTensor<OUTPUT>(2)->supportArgType(ADD_TO, {ASSIGN_TO, ADD_TO});

setShapeInferer<MulAttribute>([](std::vector<topology::TensorPtr>& ins,
                                 std::vector<topology::TensorPtr>& outs,
                                 const MulAttribute& attr) -> Error {
  topology::TensorPtr& in0 = ins[0];
  topology::TensorPtr& in1 = ins[1];
  topology::TensorPtr& out = outs[0];

  auto err = MatrixShape(in0->shape(), in1->shape(), attr, *out);
  if (!err.isOK()) return err;
  out->setDataType(topology::DataType::DENSE);
  out->setSequenceType(ins[0]->sequenceType());

  constexpr uint8_t kSparse = 0x01;
  constexpr uint8_t kDense = 0x02;

  uint8_t in0Type =
      in0->dataType() == topology::DataType::DENSE ? kDense : kSparse;
  uint8_t in1Type =
      in1->dataType() == topology::DataType::DENSE ? kDense : kSparse;
  uint8_t inputType = (in0Type << 4) | in1Type;

  if (inputType ==
      ((kSparse << 4) | kSparse)) {  // sparse matrix * sparse matrix
    return Error("Not support sparse*sparse matrix");
  } else if (inputType == ((kDense << 4) | kDense)) {
    // dense matrix * dense matrix
    return Error();
  } else if (inputType == ((kDense << 4) | kSparse)) {
    if (attr.aTrans) {
      return Error("dense*sparse not support Matrix A is transpose");
    }
    return Error();
  } else if (inputType == ((kSparse << 4) | kDense)) {
    if (attr.bTrans) {
      return Error("sparse*dense not support Matrix B is transpose");
    }
    if (in0->sparseFormatType() != topology::SparseDataFormat::SPARSE_CSR) {
      return Error("sparse*dense only supported SPARSE_CSR format");
    }
    return Error();
  } else {
    return Error("Unexpected branch.");
  }
});

END_REGISTER_FUNCTION(Mul)

template <DeviceType Device>
static Error mulToSparse(const BufferArgs& inputs,
                         const BufferArgs& outputs,
                         const MulAttribute& attr) {
  real scaleT = (outputs[0].getArgType() == ADD_TO) ? 1.0 : 0.0;
  /// sparse matrix = dense matrix * dense matrix
  auto outSparseMat = outputs[0].sparse().SparseMatrix<Device>();
  if (!inputs[0].isSparseArg() && !inputs[1].isSparseArg() &&
      outputs[0].isSparseArg()) {
    MulOp<Device>(outSparseMat,
                  inputs[0].matrix<Device>(),
                  inputs[1].matrix<Device>(),
                  1.0,  // scaleAB
                  scaleT,
                  attr.aTrans,
                  attr.bTrans);
  }
  return Error();
}

BEGIN_REGISTER_FUNCTION(MulToSparse, mulToSparse, MulAttribute)
addTensor<INPUT>(2);
addTensor<INPUT>(2);
addTensor<OUTPUT>(2, -1, {topology::DataType::SPARSE})
    ->supportArgType(ADD_TO, {ASSIGN_TO, ADD_TO});
setShapeInferer<MulAttribute>([](std::vector<topology::TensorPtr>& ins,
                                 std::vector<topology::TensorPtr>& outs,
                                 const MulAttribute& attr) -> Error {
  topology::TensorPtr& in0 = ins[0];
  topology::TensorPtr& in1 = ins[1];
  topology::TensorPtr& out = outs[0];

  auto err = MatrixShape(in0->shape(), in1->shape(), attr, *out);
  if (!err.isOK()) return err;
  out->setSequenceType(in0->sequenceType());
  out->setDataType(topology::DataType::SPARSE);
  return Error();
});
END_REGISTER_FUNCTION(MulToSparse)
}  // namespace paddle
