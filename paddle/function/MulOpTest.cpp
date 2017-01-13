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
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT

void testSpMatrixMul(int M, int N, int K, real rate, real scale1, real scale2) {
  /// todo(tianbing) check CPU/GPU
  const auto gpuFunc = FunctionBase::funcRegistrar_.createByType("MulOP-GPU");
  gpuFunc->init(FuncConfig().set("scaleAB", scale1).set("scaleT", scale2));

  int nnz = M * K * rate;
  auto gpuA = std::make_shared<GpuSparseMatrix>(M, K, nnz);
  const auto gpuB = std::make_shared<GpuMatrix>(K, N);
  const auto gpuOut = std::make_shared<GpuMatrix>(M, N);

  gpuA->randomizeUniform();
  gpuB->randomizeUniform();
  gpuOut->randomizeUniform();

  BufferArgs inputs;
  BufferArgs outputs;
  inputs.addArg(*gpuA);
  inputs.addArg(*gpuB);
  outputs.addArg(*gpuOut);

  gpuFunc->calc(inputs, outputs);
}

TEST(SMatrix, sMatrixMul) {
  for (auto M : {1, 40, 128, 200}) {
    for (auto N : {100, 2000, 20480}) {
      for (auto K : {100, 512, 1024}) {
        /// todo(tianbing), add scaleAB and scaleT
        VLOG(3) << " M=" << M << " N=" << N << " K=" << K;
        testSpMatrixMul(M, N, K, 0.05, 1, 1);
      }
    }
  }
}
