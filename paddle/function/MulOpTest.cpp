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

#include <gtest/gtest.h>
#include "FunctionTest.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/SparseMatrix.h"
#include "paddle/math/tests/test_matrixUtil.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT

/**
 *  C += A * B, A, B, C dense matrix
 *  dense = dense * dense
 */
void testFuncDDDMatrix(
    bool transa, bool transb, size_t dimM, size_t dimN, size_t dimK) {
  real scaleT = 1.0;
  size_t heightA = (transa == false) ? dimM : dimK;
  size_t widthA = (transa == false) ? dimK : dimM;
  size_t heightB = (transb == false) ? dimK : dimN;
  size_t widthB = (transb == false) ? dimN : dimK;
  size_t heightC = dimM;
  size_t widthC = dimN;
  // init Test object
  CpuGpuFuncCompare test(
      "MulOp", FuncConfig().set("aTrans", transa).set("bTrans", transb));
  // prepare input arguments
  /// matrix A : HA * WA
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{heightA, widthA}));
  /// matrix B: HB * WB
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{heightB, widthB}));

  /// output matrix C: HC * WC
  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{heightC, widthC}),
                  scaleT == 1.0 ? ADD_TO : ASSIGN_TO);
  // run Function
  test.run();
}

TEST(MulOp, DDDMatrixMul) {
  LOG(INFO) << "function test for dense = dense * dense matrix";
  for (const auto transa : {false, true}) {
    for (const auto transb : {false, true}) {
      for (const auto dimM : {1, 10, 100}) {
        for (const auto dimN : {1, 10}) {
          for (const auto dimK : {8}) {
            if (transa && transb) {
              continue;
            }
            VLOG(3) << std::setiosflags(std::ios::left) << std::setfill(' ')
                    << " transa=" << transa << " transb=" << transb
                    << " dimM=" << std::setw(5) << dimM
                    << " dimN=" << std::setw(5) << dimN
                    << " dimK=" << std::setw(5) << dimK;
            testFuncDDDMatrix(transa, transb, dimM, dimN, dimK);
          }
        }
      }
    }
  }
}

/**
 * C += A * B, B, C dense, A sparse
 * dense = sparse * dense
 */
void testFuncDSparseDMatrix(
    size_t dimM, size_t dimN, size_t dimK, size_t nnz, SparseFormat FORMAT) {
  real scaleT = 1.0;
  // init Test object
  CpuGpuFuncCompare test(
      "MulOp", FuncConfig().set("aTrans", false).set("bTrans", false));
  // prepare input arguments
  /// sparse matrix A : M * K
  test.addInputs(SparseMatrixArg(
      VALUE_TYPE_FLOAT, TensorShape{dimM, dimK}, nnz, FORMAT, FLOAT_VALUE));
  /// matrix B: K * N
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{dimK, dimN}));

  /// output matrix C: M * N
  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{dimM, dimN}),
                  scaleT == 1.0 ? ADD_TO : ASSIGN_TO);
  // run Function
  test.run();
}

TEST(MuLOp, DSparseDMul) {
  LOG(INFO) << "function test for dense = sparse * dense matrix";
  for (const auto dimM : {10, 100, 1000}) {
    for (const auto dimN : {10, 100}) {
      for (const auto dimK : {3, 10}) {
        for (const auto nnz : {3, 10}) {
          for (const auto FORMAT : {SPARSE_CSR}) {
            VLOG(3) << std::setiosflags(std::ios::left) << std::setfill(' ')
                    << " dimM=" << std::setw(5) << dimM
                    << " dimN=" << std::setw(5) << dimN
                    << " dimK=" << std::setw(5) << dimK
                    << " nnz=" << std::setw(5) << nnz
                    << " format=" << std::setw(5) << FORMAT;
            testFuncDSparseDMatrix(dimM, dimN, dimK, nnz, FORMAT);
          }
        }
      }
    }
  }
}

/**
 * C += A * B, A, C dense, B sparse
 * dense = dense * sparse
 */
void testFuncDDSparseMatrix(
    size_t dimM, size_t dimN, size_t dimK, size_t nnz, SparseFormat FORMAT) {
  real scaleT = 1.0;
  // init Test object
  CpuGpuFuncCompare test(
      "MulOp", FuncConfig().set("aTrans", false).set("bTrans", false));
  // prepare input arguments
  /// matrix A : M * K
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{dimM, dimK}));

  /// matrix B: K * N
  test.addInputs(SparseMatrixArg(
      VALUE_TYPE_FLOAT, TensorShape{dimK, dimN}, nnz, FORMAT, FLOAT_VALUE));

  /// output matrix C: M * N
  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{dimM, dimN}),
                  scaleT == 1.0 ? ADD_TO : ASSIGN_TO);
  // run Function
  test.run();
}

TEST(MulOp, DDSparseMul) {
  LOG(INFO) << "function test for dense = dense * sparse matrix";
  for (const auto dimM : {10, 100, 1000}) {
    for (const auto dimN : {10, 100}) {
      for (const auto dimK : {3, 10}) {
        for (const auto nnz : {3, 10}) {
          for (const auto FORMAT : {SPARSE_CSR, SPARSE_CSC}) {
            VLOG(3) << std::setiosflags(std::ios::left) << std::setfill(' ')
                    << " dimM=" << std::setw(5) << dimM
                    << " dimN=" << std::setw(5) << dimN
                    << " dimK=" << std::setw(5) << dimK
                    << " nnz=" << std::setw(5) << nnz
                    << " format=" << std::setw(5) << FORMAT;
            testFuncDDSparseMatrix(dimM, dimN, dimK, nnz, FORMAT);
          }
        }
      }
    }
  }
}

/**
 * C += A * B, A sparse, B, C dense
 * sparse = dense * dense
 */
void testFuncSparseDDMatrix(
    size_t dimM, size_t dimN, size_t dimK, size_t nnz, SparseFormat FORMAT) {
  real scaleT = 1.0;
  // init Test object
  CpuGpuFuncCompare test(
      "MulOp", FuncConfig().set("aTrans", false).set("bTrans", false));
  // prepare input arguments
  /// matrix A : M * K
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{dimM, dimK}));

  /// matrix B: K * N
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{dimK, dimN}));

  /// output sparse matrix C: M * N
  test.addOutputs(
      SparseMatrixArg(
          VALUE_TYPE_FLOAT, TensorShape{dimM, dimN}, nnz, FORMAT, FLOAT_VALUE),
      scaleT == 1.0 ? ADD_TO : ASSIGN_TO);
  // run Function
  test.run();
}

TEST(MulOp, SparseDDMul) {
  LOG(INFO) << "function test for sparse = dense * dense matrix";
  for (const auto dimM : {10, 100, 1000}) {
    for (const auto dimN : {10, 100}) {
      for (const auto dimK : {3, 10}) {
        for (const auto nnz : {3, 10}) {
          for (const auto FORMAT : {SPARSE_CSC, SPARSE_CSR}) {
            VLOG(3) << std::setiosflags(std::ios::left) << std::setfill(' ')
                    << " dimM=" << std::setw(5) << dimM
                    << " dimN=" << std::setw(5) << dimN
                    << " dimK=" << std::setw(5) << dimK
                    << " nnz=" << std::setw(5) << nnz
                    << " format=" << std::setw(5) << FORMAT;
            testFuncSparseDDMatrix(dimM, dimN, dimK, nnz, FORMAT);
          }
        }
      }
    }
  }
}
