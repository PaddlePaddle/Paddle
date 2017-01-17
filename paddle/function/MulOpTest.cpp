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
