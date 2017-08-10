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

#ifndef PADDLE_ONLY_CPU
/// This unittest checks GpuSparseMatrix/CpuSparseMatrix get same result,
//  so disable when
/// only cpu version.

#include <gtest/gtest.h>
#include "paddle/math/Matrix.h"
#include "paddle/utils/Util.h"
#include "test_matrixUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

static inline int uniformRandom(int n) { return n == 0 ? 0 : rand() % n; }

void testSpMatrixAddBias(int M, int N, real rate, real scale) {
  int nnz = M * N * rate;

  MatrixPtr cpuA(new CpuSparseMatrix(M, N, nnz));
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(1, N);

  MatrixPtr gpuA(new GpuSparseMatrix(M, N, nnz));
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(1, N);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();

  hl_stream_t stream(HPPL_STREAM_1);
  gpuA->copyFrom(*cpuA, stream);
  gpuB->copyFrom(*cpuB, stream);
  hl_stream_synchronize(stream);

  cpuA->addBias(*cpuB, scale);
  gpuA->addBias(*gpuB, scale);

  MatrixPtr outputCheck(new CpuSparseMatrix(M, N, nnz));
  outputCheck->copyFrom(*gpuA, stream);
  hl_stream_synchronize(stream);
  checkSMatrixEqual2(std::dynamic_pointer_cast<CpuSparseMatrix>(cpuA),
                     std::dynamic_pointer_cast<CpuSparseMatrix>(outputCheck));
}

void testSpMatrixAddDense(int M, int N, real rate) {  // add3
  int nnz = M * N * rate;

  MatrixPtr cpuA(new CpuSparseMatrix(M, N, nnz));
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(M, N);

  MatrixPtr gpuA(new GpuSparseMatrix(M, N, nnz));
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(M, N);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();

  hl_stream_t stream(HPPL_STREAM_3);
  gpuA->copyFrom(*cpuA, stream);
  gpuB->copyFrom(*cpuB, stream);
  hl_stream_synchronize(stream);

  cpuA->add3(cpuB);
  gpuA->add3(gpuB);

  MatrixPtr outputCheck(new CpuSparseMatrix(M, N, nnz));
  outputCheck->copyFrom(*gpuA, stream);
  hl_stream_synchronize(stream);
  checkSMatrixEqual2(std::dynamic_pointer_cast<CpuSparseMatrix>(cpuA),
                     std::dynamic_pointer_cast<CpuSparseMatrix>(outputCheck));
}

void testSpMatrixMul(int M, int N, int K, real rate) {
  int nnz = M * N * rate;

  MatrixPtr cpuA = std::make_shared<CpuMatrix>(M, K);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(N, K);
  MatrixPtr cpuC(new CpuSparseMatrix(M, N, nnz));

  MatrixPtr gpuA = std::make_shared<GpuMatrix>(M, K);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(N, K);
  MatrixPtr gpuC(new GpuSparseMatrix(M, N, nnz));

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  cpuC->randomizeUniform();

  hl_stream_t stream(HPPL_STREAM_3);
  gpuA->copyFrom(*cpuA, stream);
  gpuB->copyFrom(*cpuB, stream);
  gpuC->copyFrom(*cpuC, stream);
  hl_stream_synchronize(stream);

  cpuC->mul(*cpuA, *cpuB->getTranspose(), 1, 1);
  gpuC->mul(*gpuA, *gpuB->getTranspose(), 1, 1);

  MatrixPtr outputCheck(new CpuSparseMatrix(M, N, nnz));
  outputCheck->copyFrom(*gpuC, stream);
  hl_stream_synchronize(stream);
  checkSMatrixErr(std::dynamic_pointer_cast<CpuSparseMatrix>(cpuC),
                  std::dynamic_pointer_cast<CpuSparseMatrix>(outputCheck));
}

void testSpMatrixCollectBias(int M, int N, real rate) {
  int nnz = M * N * rate;
  LOG(INFO) << "nnz=" << nnz;

  MatrixPtr cpuA(new CpuSparseMatrix(M, N, nnz));
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(1, N);

  MatrixPtr gpuA(new GpuSparseMatrix(M, N, nnz));
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(1, N);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();

  hl_stream_t stream(HPPL_STREAM_3);
  gpuA->copyFrom(*cpuA, stream);
  gpuB->copyFrom(*cpuB, stream);
  hl_stream_synchronize(stream);

  cpuB->collectBias(*cpuA, 1);
  gpuB->collectBias(*gpuA, 1);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(1, N);
  outputCheck->copyFrom(*gpuB, stream);
  hl_stream_synchronize(stream);
  checkMatrixErr(*cpuB, *outputCheck);
}

TEST(SMatrix, sMatrixOp) {
  for (auto height : {1, 11, 200}) {
    for (auto width : {200, 2048, 20480}) {
      VLOG(3) << " height=" << height << " width=" << width;
      for (auto rate : {0.02, 0.1}) {
        testSpMatrixAddDense(height, width, rate);
        testSpMatrixAddBias(height, width, rate, 1.0);
      }
    }
  }
}

TEST(SMatrix, sMatrixMul) {
  for (auto M : {1, 40, 128, 200}) {
    for (auto N : {100, 2000, 20480}) {
      for (auto K : {100, 512, 1024}) {
        VLOG(3) << " M=" << M << " N=" << N << " K=" << K;
        testSpMatrixMul(M, N, K, 0.05);
      }
    }
  }
}

TEST(SMatrix, sMatrixCollectBias) {
  for (auto height : {1, 128, 200}) {
    for (auto width : {100, 2048, 20480}) {
      VLOG(3) << " height=" << height << " width=" << width;
      testSpMatrixCollectBias(height, width, 0.1);
    }
  }
}

#endif
