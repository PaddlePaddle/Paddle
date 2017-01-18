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

#include <gtest/gtest.h>
/// todo(tianbing), delete
#include <iostream>
#include "FunctionTest.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/SparseMatrix.h"
#include "paddle/math/tests/test_matrixUtil.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT

/**
 *  C = alpha * C + beta * (A * B)
 */
void testMatrixMul(bool transa, bool transb, int dimM, int dimN, int dimK) {
  real alpha = 1.5;
  real beta = 2.0;

  const auto cpuFunc = FunctionBase::funcRegistrar_.createByType("MulOp-CPU");
  cpuFunc->init(FuncConfig().set("scaleAB", alpha).set("scaleT", beta));
  const auto gpuFunc = FunctionBase::funcRegistrar_.createByType("MulOp-GPU");
  gpuFunc->init(FuncConfig().set("scaleAB", alpha).set("scaleT", beta));

  int heightA = (transa == false) ? dimM : dimK;
  int widthA = (transa == false) ? dimK : dimM;
  int heightB = (transb == false) ? dimK : dimN;
  int widthB = (transb == false) ? dimN : dimK;
  int heightC = dimM;
  int widthC = dimN;

  auto cpuA = std::make_shared<CpuMatrix>(heightA, widthA, transa);
  auto cpuB = std::make_shared<CpuMatrix>(heightB, widthB, transb);
  auto cpuC = std::make_shared<CpuMatrix>(heightC, widthC);
  auto gpuA = std::make_shared<GpuMatrix>(heightA, widthA, transa);
  auto gpuB = std::make_shared<GpuMatrix>(heightB, widthB, transb);
  auto gpuC = std::make_shared<GpuMatrix>(heightC, widthC);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  cpuC->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  gpuC->copyFrom(*cpuC);

  BufferArgs cpuInputs;
  BufferArgs cpuOutputs;
  cpuInputs.addArg(*cpuA);
  cpuInputs.addArg(*cpuB);
  cpuOutputs.addArg(*cpuC, ADD_TO);
  cpuFunc->calc(cpuInputs, cpuOutputs);

  BufferArgs gpuInputs;
  BufferArgs gpuOutputs;
  gpuInputs.addArg(*gpuA);
  gpuInputs.addArg(*gpuB);
  gpuOutputs.addArg(*gpuC, ADD_TO);
  gpuFunc->calc(gpuInputs, gpuOutputs);

  autotest::TensorCheckErr(*cpuC, *gpuC);
}

TEST(Matrix, mul) {
  LOG(INFO) << "test for dense = dense * dense matrix";
  for (auto transa : {false, true}) {
    for (auto transb : {false, true}) {
      for (auto dimM : {1, 10, 100}) {
        for (auto dimN : {1, 10}) {
          for (auto dimK : {8}) {
            if (true == transa && true == transb) {
              continue;
            }
            VLOG(3) << setiosflags(std::ios::left) << std::setfill(' ')
                    << " transa=" << transa << " transb=" << transb
                    << " dimM=" << std::setw(5) << dimM
                    << " dimN=" << std::setw(5) << dimN
                    << " dimK=" << std::setw(5) << dimK;

            testMatrixMul(transa, transb, dimM, dimN, dimK);
          }
        }
      }
    }
  }
}

struct MatrixPara {
  size_t height;
  size_t width;
  bool trans;
  bool sparse;
  size_t nnz;
  SparseFormat format;
};

/**
  * C += A * B, A, C dense, B sparse
  */
void testDSparseDMatrix() {
  real alpha = 1.0;
  real beta = 1.0;
  const auto cpuFunc = FunctionBase::funcRegistrar_.createByType("MulOp-CPU");
  cpuFunc->init(FuncConfig().set("scaleAB", alpha).set("scaleT", beta));
  const auto gpuFunc = FunctionBase::funcRegistrar_.createByType("MulOp-GPU");
  gpuFunc->init(FuncConfig().set("scaleAB", alpha).set("scaleT", beta));

  constexpr size_t dimM = 2;
  constexpr size_t dimN = 2;
  constexpr size_t dimK = 3;
  constexpr size_t NNZ = 3;
  constexpr SparseFormat FORMAT = SPARSE_CSC;

  MatrixPara paraA{dimM, dimK, /*trans*/ false, /*sparse*/ false, NNZ, FORMAT};
  MatrixPara paraB{dimK, dimN, /*trans*/ false, /*sparse*/ true, NNZ, FORMAT};
  MatrixPara paraC{dimM, dimN, /*trans*/ false, /*sparse*/ false, NNZ, FORMAT};

  auto cpuMatrixA =
      Matrix::create(paraA.height, paraA.width, paraA.trans, false);
  auto gpuMatrixA =
      Matrix::create(paraA.height, paraA.width, paraA.trans, true);
  auto cpuDenseA =
      Matrix::create(paraA.height, paraA.width, paraA.trans, false);
  CpuSparseMatrix cpuMatrixB(paraB.height,
                             paraB.width,
                             paraB.nnz,
                             FLOAT_VALUE,
                             paraB.format,
                             paraB.trans);

  GpuSparseMatrix gpuMatrixB(paraB.height,
                             paraB.width,
                             paraB.nnz,
                             FLOAT_VALUE,
                             paraB.format,
                             paraB.trans);

  auto cpuDenseB =
      Matrix::create(paraB.height, paraB.width, paraB.trans, false);
  auto cpuMatrixC =
      Matrix::create(paraC.height, paraC.width, paraC.trans, false);
  auto gpuMatrixC =
      Matrix::create(paraC.height, paraC.width, paraC.trans, true);
  auto cpuDenseC =
      Matrix::create(paraC.height, paraC.width, paraC.trans, false);
  auto gpuMatrixC_d2h =
      Matrix::create(paraC.height, paraC.width, paraC.trans, false);
  /*matrix init*/
  hl_stream_t stream(HPPL_STREAM_1);
  cpuMatrixA->randomizeUniform();
  cpuMatrixB.randomizeUniform();
  cpuMatrixC->randomizeUniform();

  gpuMatrixA->copyFrom(*cpuMatrixA, stream);
  gpuMatrixB.copyFrom(cpuMatrixB, stream);
  gpuMatrixC->copyFrom(*cpuMatrixC, stream);

  cpuDenseA->copyFrom(*cpuMatrixA);
  cpuDenseB->copyFrom(cpuMatrixB);
  cpuDenseC->copyFrom(*cpuMatrixC);
  hl_stream_synchronize(stream);

  LOG(INFO) << "cpuMatrixA: ";
  cpuMatrixA->print(std::cout);
  LOG(INFO) << "cpuMatrixB: ";
  (&cpuMatrixB)->print(std::cout);
  LOG(INFO) << "cpuMatrixC: ";
  cpuMatrixC->print(std::cout);

  LOG(INFO) << "cpuDenseA: ";
  cpuDenseA->print(std::cout);
  LOG(INFO) << "cpuDenseB: ";
  cpuDenseB->print(std::cout);
  LOG(INFO) << "cpuDenseC: ";
  cpuDenseC->print(std::cout);

  LOG(INFO) << "gpuMatrixA: ";
  gpuMatrixA->print(std::cout);
  LOG(INFO) << "gpuMatrixB: ";
  (&gpuMatrixB)->print(std::cout);
  LOG(INFO) << "gpuMatrixC: ";
  gpuMatrixC->print(std::cout);

  /*matrix mul*/
  BufferArgs cpuInputs;
  BufferArgs cpuOutputs;
  cpuInputs.addArg(*cpuMatrixA);
  cpuInputs.addArg(cpuMatrixB);
  cpuOutputs.addArg(*cpuMatrixC, ADD_TO);
  cpuFunc->calc(cpuInputs, cpuOutputs);

  BufferArgs gpuInputs;
  BufferArgs gpuOutputs;
  gpuInputs.addArg(*gpuMatrixA);
  gpuInputs.addArg(gpuMatrixB);
  gpuOutputs.addArg(*gpuMatrixC, ADD_TO);
  gpuFunc->calc(gpuInputs, gpuOutputs);

  BufferArgs denseInputs;
  BufferArgs denseOutputs;
  denseInputs.addArg(*cpuDenseA);
  denseInputs.addArg(*cpuDenseB);
  denseOutputs.addArg(*cpuDenseC, ADD_TO);
  cpuFunc->calc(denseInputs, denseOutputs);

  gpuMatrixC_d2h->copyFrom(*gpuMatrixC, stream);
  hl_stream_synchronize(stream);
  /*check result*/
  // autotest::TensorCheckErr(*cpuMatrixC, *gpuMatrixC);
  checkMatrixEqual(cpuMatrixC, cpuDenseC);
  checkMatrixEqual(cpuMatrixC, gpuMatrixC_d2h);
}

TEST(Matrix, SparseMatrixMul) {
  LOG(INFO) << "test for dense = dense * sparse matrix";
  testDSparseDMatrix();
}
