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

#include "Matrix.h"
#include "hl_gpu.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Util.h"

namespace paddle {

namespace {

struct SimpleCode {
  SimpleCode(size_t code, size_t numClasses) : c_(code + numClasses) {}
  inline size_t calcIndex(int bit) const { return (c_ >> (bit + 1)) - 1; }
  inline bool calcBit(int bit) const { return c_ & (1 << bit); }
  inline int getLength() const { return findLastSet(c_) - 1; }

private:
  size_t c_;
};

struct SimpleCodeTable {
  explicit SimpleCodeTable(size_t numClasses) : numClasses_(numClasses) {}
  SimpleCode operator()(size_t code) const {
    return SimpleCode(code, numClasses_);
  }
  size_t size() const { return numClasses_; }
  int getMaxCodeLength() const { return findLastSet(numClasses_ - 1); }

private:
  size_t numClasses_;
  int maxCodeLength_;
};

}  // namespace

/**
 * CodeTable class should support 3 functions:
 *
 * size_t size()
 *   return the number of codes
 *
 * int getMaxCodeLength()
 *   return the maximal code length
 *
 * Code operator()(size_t i)
 *   return the i-th code. Code class is descriebed below.
 *
 * Code class should support 3 functions:
 *
 * int getLength()
 *   return the length of the code
 *
 * bool calcIndex(int bit)
 *   bit ranges from 0 to getLength() - 1
 *   return the index for the (1+bit) level parent
 *
 * bool calcBit(int bit)
 *   return true if the bit level parent is the right child of (1+bit) level
 *   parent
 *
 */

/*
   for i:
     for j < codeLength:
       op(tmat(i, j), vec(0, index(i, j)))
*/
template <class CodeTable, class Op, class TMat, class Mat>
static void addByBitCodeT(
    Op op, CodeTable codeTable, const IVector& codes, TMat& tmat, Mat& vec) {
  CHECK(!vec.useGpu());

  size_t numClasses = codeTable.size();
  size_t maxCodeLength = codeTable.getMaxCodeLength();
  size_t numSamples = tmat.getHeight();
  size_t oWidth = tmat.getWidth();
  CHECK_EQ(tmat.getWidth(), maxCodeLength);
  CHECK_EQ(codes.getSize(), numSamples);
  CHECK_EQ(vec.getHeight(), (size_t)1);
  CHECK_EQ(vec.getWidth(), numClasses - 1);

  auto data = tmat.getData();
  auto v = vec.getData();
  const int* c = codes.getData();
  for (size_t i = 0; i < numSamples; ++i) {
    auto code = codeTable(c[i]);
    int codeLength = code.getLength();
    for (int j = 0; j < codeLength; ++j) {
      size_t index = code.calcIndex(j);
      op(data[i * oWidth + j], v[index]);
    }
  }
}

/* For j < codeLength:
   this(i, j) += vec(0, index(i, j))
*/
void CpuMatrix::addByBitCode(size_t numClasses,
                             const IVector& codes,
                             const Matrix& vec) {
  auto op = [](real& t, real v) { t += v; };
  addByBitCodeT(op, SimpleCodeTable(numClasses), codes, *this, vec);
}

/* For j < codeLength:
   vec(0, index(i, j)) += this(i, j)
*/
void CpuMatrix::addByBitCodeBackward(size_t numClasses,
                                     const IVector& codes,
                                     Matrix& vec) {
  auto op = [](real t, real& v) { v += t; };
  addByBitCodeT(op, SimpleCodeTable(numClasses), codes, *this, vec);
}

/*
  for i:
    for j < codeLength:
      op(tmat(i, j), mat.row(index(i, j)), input.row(i))
*/
template <class Op,
          class CodeTable,
          class IVec,
          class TMat,
          class WMat,
          class InMat>
void mulByBitCodeT(Op op,
                   CodeTable codeTable,
                   IVec& codes,
                   TMat& tmat,
                   WMat& weight,
                   InMat& input) {
  CHECK(!tmat.useGpu() && !weight.useGpu() && !input.useGpu());

  size_t numClasses = codeTable.size();
  size_t maxCodeLength = codeTable.getMaxCodeLength();
  size_t numSamples = tmat.getHeight();
  size_t inputDim = input.getWidth();
  size_t oWidth = tmat.getWidth();
  CHECK_EQ(tmat.getWidth(), maxCodeLength);
  CHECK_EQ(codes.getSize(), numSamples);
  CHECK_EQ(input.getHeight(), numSamples);
  CHECK_EQ(weight.getHeight(), numClasses - 1);
  CHECK_EQ(weight.getWidth(), inputDim);

  real* data = tmat.getData();
  const int* c = codes.getData();
  for (size_t i = 0; i < numSamples; ++i) {
    auto code = codeTable(c[i]);
    int codeLength = code.getLength();
    for (int j = 0; j < codeLength; ++j) {
      size_t index = code.calcIndex(j);
      op(data[i * oWidth + j], weight.rowBuf(index), input.rowBuf(i), inputDim);
    }
  }
}

/* For j < codeLength:
   this(i, j) += <weight.row(index(i, j)), input.row(i)>
*/
void CpuMatrix::mulByBitCode(size_t numClasses,
                             const IVector& codes,
                             const Matrix& weight,
                             const Matrix& input) {
  auto op = [](
      real& t, const real* weightRow, const real* inputRow, size_t inputDim) {
    real sum = 0;
    for (size_t k = 0; k < inputDim; ++k) {
      sum += weightRow[k] * inputRow[k];
    }
    t += sum;
  };

  mulByBitCodeT(op, SimpleCodeTable(numClasses), codes, *this, weight, input);
}

/* For index(i, j) >= 0:
   weight.row(index(i, j)) += this(i, j) * input.row(i)
*/
void CpuMatrix::mulByBitCodeBackwardWeight(size_t numClasses,
                                           const IVector& codes,
                                           Matrix& weight,
                                           const Matrix& input) {
  auto op = [](
      const real t, real* weightRow, const real* inputRow, size_t inputDim) {
    for (size_t k = 0; k < inputDim; ++k) {
      weightRow[k] += t * inputRow[k];
    }
  };

  mulByBitCodeT(op, SimpleCodeTable(numClasses), codes, *this, weight, input);
}

/* For j < codeLength:
   input.row(i) += this(i, j) * weight.row(index(i, j))
*/
void CpuMatrix::mulByBitCodeBackwardError(size_t numClasses,
                                          const IVector& codes,
                                          const Matrix& weight,
                                          Matrix& input) {
  auto op = [](
      const real t, const real* weightRow, real* inputRow, size_t inputDim) {
    for (size_t k = 0; k < inputDim; ++k) {
      inputRow[k] += t * weightRow[k];
    }
  };

  mulByBitCodeT(op, SimpleCodeTable(numClasses), codes, *this, weight, input);
}

template <class CodeTable>
void sumByBitCodeT(CodeTable codeTable,
                   IVector& codes,
                   const CpuMatrix& tmat,
                   Matrix& sum,
                   real scaleSum) {
  size_t maxCodeLength = codeTable.getMaxCodeLength();
  size_t numSamples = tmat.getHeight();
  size_t oWidth = tmat.getWidth();
  CHECK_EQ(tmat.getWidth(), maxCodeLength);
  CHECK_EQ(codes.getSize(), numSamples);
  CHECK_EQ(sum.getHeight(), numSamples);
  CHECK_EQ(sum.getWidth(), (size_t)1);

  const real* data = tmat.getData();
  real* s = sum.getData();
  int* c = codes.getData();
  for (size_t i = 0; i < numSamples; ++i) {
    real sm = 0;
    auto code = codeTable(c[i]);
    int codeLength = code.getLength();
    for (int j = 0; j < codeLength; ++j) {
      if (code.calcBit(j)) {
        sm += data[i * oWidth + j];
      }
    }
    s[i] = scaleSum * sm;
  }
}

/* For j < codeLength:
   sum(i, 0) = \sum_j  bit(i, j) * this(i, j)
*/
void CpuMatrix::sumByBitCode(size_t numClasses,
                             IVector& codes,
                             Matrix& sum,
                             real scaleSum) {
  sumByBitCodeT(SimpleCodeTable(numClasses), codes, *this, sum, scaleSum);
}

template <class CodeTable>
void subByBitCodeT(CodeTable codeTable, IVector& codes, CpuMatrix& tmat) {
  size_t maxCodeLength = codeTable.getMaxCodeLength();
  size_t numSamples = tmat.getHeight();
  size_t oWidth = tmat.getWidth();
  CHECK_EQ(tmat.getWidth(), maxCodeLength);
  CHECK_EQ(codes.getSize(), numSamples);

  real* data = tmat.getData();
  int* c = codes.getData();
  for (size_t i = 0; i < numSamples; ++i) {
    auto code = codeTable(c[i]);
    int codeLength = code.getLength();
    for (int j = 0; j < codeLength; ++j) {
      if (code.calcBit(j)) {
        data[i * oWidth + j] -= 1;
      }
    }
  }
}

/* For j < codeLength
   this(i, j) -= bit(i, j)
*/
void CpuMatrix::subByBitCode(size_t numClasses, IVector& codes) {
  subByBitCodeT(SimpleCodeTable(numClasses), codes, *this);
}

}  // namespace paddle
