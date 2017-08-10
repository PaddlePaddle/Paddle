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

#include "hl_batch_transpose.h"
#include "test_matrixUtil.h"

using namespace paddle;  // NOLINT

#ifndef PADDLE_ONLY_CPU
TEST(MatrixBatchTransTest, test_batch_matrix_transpose) {
  const int nx = 100;
  const int ny = 50;
  const int numSamples = 50;

  MatrixPtr cMat = Matrix::create(numSamples, nx * ny, false, false);
  MatrixPtr gMat = Matrix::create(numSamples, nx * ny, false, true);

  MatrixPtr cBatchTransMat = Matrix::create(numSamples, nx * ny, false, false);
  MatrixPtr gBatchTransMat = Matrix::create(numSamples, nx * ny, false, true);
  MatrixPtr cMat_d2h = Matrix::create(numSamples, nx * ny, false, false);

  real* cData = cMat->getData();
  real* gold = cBatchTransMat->getData();

  // host
  for (int sample_id = 0; sample_id < numSamples; ++sample_id)
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
        cData[sample_id * nx * ny + j * nx + i] = j * nx + i;

  // correct result for error checking
  for (int sample_id = 0; sample_id < numSamples; ++sample_id)
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
        gold[sample_id * nx * ny + i * ny + j] =
            cData[sample_id * nx * ny + j * nx + i];
  // device
  gMat->copyFrom(*cMat, HPPL_STREAM_DEFAULT);
  batchTranspose(
      gMat->getData(), gBatchTransMat->getData(), nx, ny, numSamples);
  cMat_d2h->copyFrom(*gBatchTransMat, HPPL_STREAM_DEFAULT);
  checkMatrixEqual(cBatchTransMat, cMat_d2h);
}
#endif
