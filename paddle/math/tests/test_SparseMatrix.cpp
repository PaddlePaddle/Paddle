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

#include <paddle/utils/PythonUtil.h>
#include <vector>
#include "test_matrixUtil.h"

using namespace paddle;  // NOLINT

TEST(Matrix, CopyCpuMatrixToSparseMatrix) {
  const size_t HEIGHT = 20;
  const size_t WIDTH = 10;
  const size_t WIDTH_TEST = 15;
  MatrixPtr testMatrix(
      new CpuSparseMatrix(HEIGHT, WIDTH, HEIGHT * 5, FLOAT_VALUE, SPARSE_CSR));
  MatrixPtr testCpuMatrix(new CpuMatrix(HEIGHT, WIDTH));
  testCpuMatrix->randomizeUniform();
  testMatrix->copyFrom(*testCpuMatrix, HPPL_STREAM_DEFAULT);
  MatrixPtr mulCpuMatrix(new CpuMatrix(WIDTH, WIDTH_TEST));
  mulCpuMatrix->randomizeUniform();
  MatrixPtr ret1(new CpuMatrix(HEIGHT, WIDTH_TEST)),
      ret2(new CpuMatrix(HEIGHT, WIDTH_TEST));
  ret1->zeroMem();
  ret2->zeroMem();
  ret1->mul(*testMatrix, *mulCpuMatrix, 1.0, 1.0);
  ret2->mul(*testCpuMatrix, *mulCpuMatrix, 1.0, 1.0);
  checkMatrixEqual(ret1, ret2);
}

struct MatrixPara {
  size_t height;
  size_t width;
  bool trans;
  bool sparse;
  size_t nnz;
  SparseFormat format;
};

#ifndef PADDLE_ONLY_CPU
void test_sparse_matrix_mul(MatrixPara paraA,
                            MatrixPara paraB,
                            MatrixPara paraC) {
  // for cpu sparse matrix mul
  MatrixPtr cpuMatrixA, cpuMatrixB, cpuMatrixC, gpuMatrixC_d2h;
  // for gpu sparse matrix mul
  MatrixPtr gpuMatrixA, gpuMatrixB, gpuMatrixC;
  // for cpu dense matrix mul
  MatrixPtr cpuDenseA, cpuDenseB, cpuDenseC;

  if (paraA.sparse) {
    cpuMatrixA = Matrix::createSparseMatrix(paraA.height,
                                            paraA.width,
                                            paraA.nnz,
                                            FLOAT_VALUE,
                                            paraA.format,
                                            paraA.trans,
                                            false);
    gpuMatrixA = Matrix::createSparseMatrix(paraA.height,
                                            paraA.width,
                                            paraA.nnz,
                                            FLOAT_VALUE,
                                            paraA.format,
                                            paraA.trans,
                                            true);
  } else {
    cpuMatrixA = Matrix::create(paraA.height, paraA.width, paraA.trans, false);
    gpuMatrixA = Matrix::create(paraA.height, paraA.width, paraA.trans, true);
  }
  cpuDenseA = Matrix::create(paraA.height, paraA.width, paraA.trans, false);

  if (paraB.sparse) {
    cpuMatrixB = Matrix::createSparseMatrix(paraB.height,
                                            paraB.width,
                                            paraB.nnz,
                                            FLOAT_VALUE,
                                            paraB.format,
                                            paraB.trans,
                                            false);
    gpuMatrixB = Matrix::createSparseMatrix(paraB.height,
                                            paraB.width,
                                            paraB.nnz,
                                            FLOAT_VALUE,
                                            paraB.format,
                                            paraB.trans,
                                            true);
  } else {
    cpuMatrixB = Matrix::create(paraB.height, paraB.width, paraB.trans, false);
    gpuMatrixB = Matrix::create(paraB.height, paraB.width, paraB.trans, true);
  }
  cpuDenseB = Matrix::create(paraB.height, paraB.width, paraB.trans, false);

  if (paraC.sparse) {
    cpuMatrixC = Matrix::createSparseMatrix(paraC.height,
                                            paraC.width,
                                            paraC.nnz,
                                            FLOAT_VALUE,
                                            paraC.format,
                                            paraC.trans,
                                            false);
    gpuMatrixC = Matrix::createSparseMatrix(paraC.height,
                                            paraC.width,
                                            paraC.nnz,
                                            FLOAT_VALUE,
                                            paraC.format,
                                            paraC.trans,
                                            true);
    gpuMatrixC_d2h = Matrix::createSparseMatrix(paraC.height,
                                                paraC.width,
                                                paraC.nnz,
                                                FLOAT_VALUE,
                                                paraC.format,
                                                paraC.trans,
                                                false);
  } else {
    cpuMatrixC = Matrix::create(paraC.height, paraC.width, paraC.trans, false);
    gpuMatrixC = Matrix::create(paraC.height, paraC.width, paraC.trans, true);
    gpuMatrixC_d2h =
        Matrix::create(paraC.height, paraC.width, paraC.trans, false);
  }
  cpuDenseC = Matrix::create(paraC.height, paraC.width, paraC.trans, false);

  /*matrix init*/
  hl_stream_t stream(HPPL_STREAM_1);
  cpuMatrixA->randomizeUniform();
  cpuMatrixB->randomizeUniform();
  cpuMatrixC->randomizeUniform();

  gpuMatrixA->copyFrom(*cpuMatrixA, stream);
  gpuMatrixB->copyFrom(*cpuMatrixB, stream);
  gpuMatrixC->copyFrom(*cpuMatrixC, stream);

  cpuDenseA->copyFrom(*cpuMatrixA);
  cpuDenseB->copyFrom(*cpuMatrixB);
  cpuDenseC->copyFrom(*cpuMatrixC);

  hl_stream_synchronize(stream);

  /*matrix mul*/
  cpuMatrixC->mul(*cpuMatrixA, *cpuMatrixB, 1.0, 1.0);
  gpuMatrixC->mul(*gpuMatrixA, *gpuMatrixB, 1.0, 1.0);
  cpuDenseC->mul(*cpuDenseA, *cpuDenseB, 1.0, 1.0);

  gpuMatrixC_d2h->copyFrom(*gpuMatrixC, stream);
  hl_stream_synchronize(stream);

  /*check result*/
  if (paraC.sparse) {
    checkSMatrixEqual(
        std::dynamic_pointer_cast<CpuSparseMatrix>(cpuMatrixC),
        std::dynamic_pointer_cast<CpuSparseMatrix>(gpuMatrixC_d2h));
    checkSMatrixEqual2Dense(
        std::dynamic_pointer_cast<CpuSparseMatrix>(cpuMatrixC),
        std::dynamic_pointer_cast<CpuMatrix>(cpuDenseC));
  } else {
    checkMatrixEqual(cpuMatrixC, gpuMatrixC_d2h);
    checkMatrixEqual(cpuMatrixC, cpuDenseC);
  }
}

TEST(Matrix, SparseMatrixMul) {
  const size_t DIM_M = 4;
  const size_t DIM_N = 4;
  const size_t DIM_K = 8;
  const size_t NNZ = 5;
  for (auto format : {SPARSE_CSC, SPARSE_CSR}) {
    std::string str_format = format == SPARSE_CSC ? "CSC" : "CSR";
    LOG(INFO) << "test dense mul " << str_format;
    test_sparse_matrix_mul(
        {DIM_M, DIM_K, /*trans*/ false, /*sparse*/ false, NNZ, format},
        {DIM_K, DIM_N, /*trans*/ false, /*sparse*/ true, NNZ, format},
        {DIM_M, DIM_N, /*trans*/ false, /*sparse*/ false, NNZ, format});

    LOG(INFO) << "test dense mul " << str_format << "  trans";
    test_sparse_matrix_mul(
        {DIM_M, DIM_K, /*trans*/ false, /*sparse*/ false, NNZ, format},
        {DIM_N, DIM_K, /*trans*/ true, /*sparse*/ true, NNZ, format},
        {DIM_M, DIM_N, /*trans*/ false, /*sparse*/ false, NNZ, format});

    LOG(INFO) << "test dense mul dense 2 " << str_format;
    test_sparse_matrix_mul(
        {DIM_M, DIM_K, /*trans*/ false, /*sparse*/ false, NNZ, format},
        {DIM_K, DIM_N, /*trans*/ false, /*sparse*/ false, NNZ, format},
        {DIM_M, DIM_N, /*trans*/ false, /*sparse*/ true, NNZ, format});

    LOG(INFO) << "test denseT mul dense 2 " << str_format;
    test_sparse_matrix_mul(
        {DIM_K, DIM_M, /*trans*/ true, /*sparse*/ false, NNZ, format},
        {DIM_K, DIM_N, /*trans*/ false, /*sparse*/ false, NNZ, format},
        {DIM_M, DIM_N, /*trans*/ false, /*sparse*/ true, NNZ, format});
  }
}

TEST(Matrix, CopySparseMatrixToGpuSparseMatrix) {
  const size_t HEIGHT = 20;
  const size_t WIDTH = 10;
  const size_t WIDTH_TEST = 15;
  MatrixPtr testMatrix(
      new CpuSparseMatrix(HEIGHT, WIDTH, HEIGHT * 2, FLOAT_VALUE, SPARSE_CSR));
  MatrixPtr testCpuMatrix(new CpuMatrix(HEIGHT, WIDTH));
  testCpuMatrix->randomizeUniform();
  testMatrix->copyFrom(*testCpuMatrix, HPPL_STREAM_DEFAULT);

  MatrixPtr testGpuMatrix = testMatrix->clone(HEIGHT, WIDTH, true);
  hl_stream_t gpuStream(HPPL_STREAM_3);
  testGpuMatrix->copyFrom(*testMatrix, gpuStream);
  hl_stream_synchronize(gpuStream);

  MatrixPtr mulCpuMatrix(new CpuMatrix(WIDTH, WIDTH_TEST));
  mulCpuMatrix->randomizeUniform();
  MatrixPtr mulGpuMatrix(new GpuMatrix(WIDTH, WIDTH_TEST));
  mulGpuMatrix->copyFrom(*mulCpuMatrix);
  MatrixPtr ret1(new CpuMatrix(HEIGHT, WIDTH_TEST));
  MatrixPtr ret2(new GpuMatrix(HEIGHT, WIDTH_TEST));
  ret1->zeroMem();
  ret2->zeroMem();
  ret1->mul(*testMatrix, *mulCpuMatrix, 1.0, 1.0);
  ret2->mul(*testGpuMatrix, *mulGpuMatrix, 1.0, 1.0);
  checkMatrixEqual(ret1, ret2);
}

#endif

TEST(Matrix, SparseMatrixTranspose) {
  for (auto height : {10, 50, 100}) {
    for (auto width : {10, 50, 100}) {
      auto nnz = height * width;
      for (auto valueType : {FLOAT_VALUE, NO_VALUE}) {
        for (auto format : {SPARSE_CSR, SPARSE_CSC}) {
          for (auto sparseRate : {0.1, 0.2, 0.5}) {
            MatrixPtr matA = Matrix::createSparseMatrix(
                height, width, size_t(nnz * sparseRate), valueType, format);
            MatrixPtr matB(new CpuSparseMatrix(
                width, height, size_t(nnz * sparseRate), valueType, format));
            matA->randomizeUniform();
            matA->transpose(matB, false);

            /*dense matrix transpose*/
            CpuMatrixPtr matC(new CpuMatrix(height, width));
            matC->copyFrom(*matA);
            MatrixPtr matD(new CpuMatrix(width, height));
            matC->transpose(matD, false);

            /*check result*/
            checkSMatrixEqual2Dense(
                std::dynamic_pointer_cast<CpuSparseMatrix>(matB),
                std::dynamic_pointer_cast<CpuMatrix>(matD));
          }
        }
      }
    }
  }
}

TEST(Matrix, CpuSparseMatrixSubMatrix) {
  const size_t HEIGHT = 10;
  const size_t WIDTH = 10;
  const size_t NNZ = HEIGHT * WIDTH;
  for (auto valueType : {FLOAT_VALUE, NO_VALUE}) {
    size_t startRow = 3;
    size_t rowNum = 2;
    real sparseRate = 0.1;
    /*sparse matrix init and get subMatrix*/
    CpuSparseMatrixPtr matA = std::make_shared<CpuSparseMatrix>(
        HEIGHT, WIDTH, size_t(NNZ * sparseRate), valueType, SPARSE_CSR);
    matA->randomizeUniform();
    CpuSparseMatrixPtr matB = std::dynamic_pointer_cast<CpuSparseMatrix>(
        matA->subMatrix(startRow, rowNum));

    int start = matA->getRows()[startRow];
    int end = matA->getRows()[startRow + rowNum];

    /*compare two matrix*/
    ASSERT_EQ(matB->getElementCnt(), size_t(end - start));
    if (valueType == FLOAT_VALUE) {
      for (size_t i = 0; i < matB->getElementCnt(); i++) {
        ASSERT_FLOAT_EQ(matB->getValue()[start + i],
                        matA->getValue()[start + i]);
      }
    }

    for (size_t i = 0; i < matB->getElementCnt(); i++) {
      ASSERT_EQ(matB->getCols()[start + i], matA->getCols()[start + i]);
    }
    for (size_t i = 0; i < rowNum; i++) {
      ASSERT_EQ(matB->getRows()[i], matA->getRows()[startRow + i]);
    }
  }
}

void sparseValid(
    int* major, int* minor, size_t nnz, size_t majorLen, size_t minorLen) {
  CHECK_EQ(nnz, size_t(major[majorLen - 1]));
  CHECK_EQ(nnz, minorLen);
  for (size_t i = 0; i < majorLen - 1; i++) {
    EXPECT_LE(major[i], major[i + 1]);
    for (int j = major[i]; j < major[i + 1] - 1; j++) {
      EXPECT_LE(minor[j], minor[j + 1]);
    }
  }
}

TEST(Matrix, CpuSparseMatrixRandUniform) {
  const size_t HEIGHT = 5;
  const size_t WIDTH = 10;
  const size_t NNZ = HEIGHT * WIDTH;
  int* major = nullptr;
  int* minor = nullptr;
  size_t majorLen = 0;
  size_t minorLen = 0;
  size_t nnz = 0;
  for (auto valueType : {NO_VALUE, FLOAT_VALUE}) {
    for (auto format : {SPARSE_CSR, SPARSE_CSC}) {
      CpuSparseMatrixPtr matA = std::make_shared<CpuSparseMatrix>(
          HEIGHT, WIDTH, size_t(NNZ * 0.1), valueType, format);
      matA->randomizeUniform();
      nnz = matA->getElementCnt();
      if (format == SPARSE_CSR) {
        majorLen = matA->getHeight() + 1;
        minorLen = matA->getElementCnt();
        major = matA->getRows();
        minor = matA->getCols();
      } else {
        majorLen = matA->getWidth() + 1;
        minorLen = matA->getElementCnt();
        major = matA->getCols();
        minor = matA->getRows();
      }
      sparseValid(major, minor, nnz, majorLen, minorLen);
    }
  }
}

TEST(Matrix, CpuSparseMatrixCopyFrom) {
  size_t height = 10;
  size_t width = 8;
  int64_t indices[11] = {0, 1, 5, 5, 9, 13, 15, 17, 19, 30, 32};
  sparse_non_value_t data[32];
  for (size_t i = 0; i < 32; i++) {
    data[i].col = ::rand() % width;
  }
  CpuSparseMatrixPtr mat = std::make_shared<CpuSparseMatrix>(
      height, width, 32, NO_VALUE, SPARSE_CSR, false);
  mat->copyFrom(indices, data);

  /*compare indices*/
  size_t sum = 0;
  CHECK_EQ(sum, size_t(mat->getRows()[0]));
  for (size_t i = 1; i < height + 1; i++) {
    sum += indices[i] - indices[i - 1];
    CHECK_EQ(sum, size_t(mat->getRows()[i]));
  }
  CHECK_EQ(mat->getElementCnt(), size_t(indices[height] - indices[0]));
  for (size_t i = 0; i < mat->getElementCnt(); i++) {
    CHECK_EQ(size_t(mat->getCols()[i]), size_t(data[i].col));
  }
}

TEST(Matrix, SparseMatrixCSRFormatTrimFrom) {
  size_t height = 10;
  size_t width = 8;
  int64_t indices[11] = {0, 1, 5, 5, 9, 13, 15, 17, 19, 27, 32};
  sparse_float_value_t data[32];
  int value[32] = {
      1,                       // row_0 : 1
      5, 3, 1, 6,              // row_1 : 4
      0, 1, 2, 3,              // row_3 : 4
      4, 5, 6, 7,              // row_4 : 4
      2, 3,                    // row_5 : 2
      3, 5,                    // row_6 : 2
      0, 1,                    // row_7 : 2
      0, 1, 2, 3, 4, 5, 6, 7,  // row_8 : 8
      2, 4, 7, 3, 1            // row_9 : 5
  };
  for (size_t i = 0; i < 32; i++) {
    data[i].col = value[i];
    data[i].value = float(value[i]);
  }
  CpuSparseMatrixPtr mat = std::make_shared<CpuSparseMatrix>(
      height, width, 32, FLOAT_VALUE, SPARSE_CSR, false);
  mat->copyFrom(indices, data);

  /*compare indices*/
  size_t sum = 0;
  CHECK_EQ(sum, size_t(mat->getRows()[0]));
  for (size_t i = 1; i < height + 1; i++) {
    sum += indices[i] - indices[i - 1];
    CHECK_EQ(sum, size_t(mat->getRows()[i]));
  }
  CHECK_EQ(mat->getElementCnt(), size_t(indices[height] - indices[0]));
  for (size_t i = 0; i < mat->getElementCnt(); i++) {
    CHECK_EQ(size_t(mat->getCols()[i]), size_t(data[i].col));
  }

  size_t trimedWidth = 4;
  int64_t trimedIndices[11] = {0, 1, 3, 3, 7, 7, 9, 10, 12, 16, 19};
  sparse_float_value_t trimedData[19];
  int trimedValue[19] = {
      1,  // row_0 : 1
      3,
      1,  // row_1 : 2
      0,
      1,
      2,
      3,  // row_3 : 4
      2,
      3,  // row_5 : 2
      3,  // row_6 : 1
      0,
      1,  // row_7 : 2
      0,
      1,
      2,
      3,  // row_8 : 4
      2,
      3,
      1  // row_9 : 3
  };
  for (size_t i = 0; i < 19; i++) {
    trimedData[i].col = trimedValue[i];
    trimedData[i].value = float(trimedValue[i]);
  }
  CpuSparseMatrixPtr matA = std::make_shared<CpuSparseMatrix>(
      height, trimedWidth, 19, FLOAT_VALUE, SPARSE_CSR, false);
  matA->copyFrom(trimedIndices, trimedData);

  /*compare indices*/
  sum = 0;
  CHECK_EQ(sum, size_t(matA->getRows()[0]));
  for (size_t i = 1; i < height + 1; i++) {
    sum += trimedIndices[i] - trimedIndices[i - 1];
    CHECK_EQ(sum, size_t(matA->getRows()[i]));
  }
  CHECK_EQ(matA->getElementCnt(),
           size_t(trimedIndices[height] - trimedIndices[0]));
  for (size_t i = 0; i < matA->getElementCnt(); i++) {
    CHECK_EQ(size_t(matA->getCols()[i]), size_t(trimedData[i].col));
  }

  CpuSparseMatrixPtr matB = std::make_shared<CpuSparseMatrix>(
      height, trimedWidth, height, FLOAT_VALUE, SPARSE_CSR, false);
  matB->trimFrom(*mat);
  checkSMatrixEqual2(matA, matB);

#ifndef PADDLE_ONLY_CPU
  GpuSparseMatrixPtr matC = std::make_shared<GpuSparseMatrix>(
      height, trimedWidth, height, FLOAT_VALUE, SPARSE_CSR, true);
  matC->trimFrom(*mat);

  CpuSparseMatrixPtr matD =
      std::make_shared<CpuSparseMatrix>(height,
                                        trimedWidth,
                                        matC->getElementCnt(),
                                        FLOAT_VALUE,
                                        SPARSE_CSR,
                                        false);
  matD->copyFrom(*matC, HPPL_STREAM_DEFAULT);
  hl_stream_synchronize(HPPL_STREAM_DEFAULT);
  checkSMatrixEqual2(matA, matD);
#endif
}

TEST(Matrix, SparseMatrixCSCFormatTrimFrom) {
  size_t height = 8;
  size_t width = 10;
  int indices[11] = {0, 1, 5, 5, 9, 13, 15, 17, 19, 27, 32};
  int value[32] = {
      1,                       // col_0 : 1
      5, 3, 1, 6,              // col_1 : 4
      0, 1, 2, 3,              // col_3 : 4
      4, 5, 6, 7,              // col_4 : 4
      2, 3,                    // col_5 : 2
      3, 5,                    // col_6 : 2
      0, 1,                    // col_7 : 2
      0, 1, 2, 3, 4, 5, 6, 7,  // col_8 : 8
      2, 4, 7, 3, 1            // col_9 : 5
  };
  std::vector<int> rows(value, value + 32);
  std::vector<int> cols(indices, indices + 11);
  std::vector<real> values(value, value + 32);
  CpuSparseMatrixPtr mat = std::make_shared<CpuSparseMatrix>(
      height, width, 32, FLOAT_VALUE, SPARSE_CSC, false);
  mat->copyFrom(rows, cols, values);

  /*compare indices*/
  size_t sum = 0;
  CHECK_EQ(sum, size_t(mat->getCols()[0]));
  for (size_t i = 1; i < width + 1; i++) {
    sum += indices[i] - indices[i - 1];
    CHECK_EQ(sum, size_t(mat->getCols()[i]));
  }
  CHECK_EQ(mat->getElementCnt(), size_t(indices[width] - indices[0]));
  for (size_t i = 0; i < mat->getElementCnt(); i++) {
    CHECK_EQ(size_t(mat->getRows()[i]), size_t(value[i]));
  }

  size_t trimedWidth = 5;
  int trimedIndices[6] = {0, 1, 5, 5, 9, 13};
  int trimedValue[13] = {
      1,  // col_0 : 1
      5,
      3,
      1,
      6,  // col_1 : 4
      0,
      1,
      2,
      3,  // col_3 : 4
      4,
      5,
      6,
      7  // col_4 : 4
  };
  std::vector<int> rowsA(trimedValue, trimedValue + 13);
  std::vector<int> colsA(trimedIndices, trimedIndices + 6);
  std::vector<real> valuesA(trimedValue, trimedValue + 13);
  CpuSparseMatrixPtr matA = std::make_shared<CpuSparseMatrix>(
      height, trimedWidth, 13, FLOAT_VALUE, SPARSE_CSC, false);
  matA->copyFrom(rowsA, colsA, valuesA);

  /*compare indices*/
  sum = 0;
  CHECK_EQ(sum, size_t(matA->getCols()[0]));
  for (size_t i = 1; i < trimedWidth + 1; i++) {
    sum += trimedIndices[i] - trimedIndices[i - 1];
    CHECK_EQ(sum, size_t(matA->getCols()[i]));
  }
  CHECK_EQ(matA->getElementCnt(),
           size_t(trimedIndices[trimedWidth] - trimedIndices[0]));
  for (size_t i = 0; i < matA->getElementCnt(); i++) {
    CHECK_EQ(size_t(matA->getRows()[i]), size_t(rowsA[i]));
  }

  CpuSparseMatrixPtr matB = std::make_shared<CpuSparseMatrix>(
      height, trimedWidth, height, FLOAT_VALUE, SPARSE_CSC, false);
  matB->trimFrom(*mat);
  checkSMatrixEqual2(matA, matB);

#ifndef PADDLE_ONLY_CPU
  GpuSparseMatrixPtr matC = std::make_shared<GpuSparseMatrix>(
      height, trimedWidth, height, FLOAT_VALUE, SPARSE_CSC, true);
  matC->trimFrom(*mat);

  CpuSparseMatrixPtr matD =
      std::make_shared<CpuSparseMatrix>(height,
                                        trimedWidth,
                                        matC->getElementCnt(),
                                        FLOAT_VALUE,
                                        SPARSE_CSC,
                                        false);
  matD->copyFrom(*matC, HPPL_STREAM_DEFAULT);
  hl_stream_synchronize(HPPL_STREAM_DEFAULT);
  checkSMatrixEqual2(matA, matD);
#endif
}
