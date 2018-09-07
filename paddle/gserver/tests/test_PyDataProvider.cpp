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

#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "paddle/gserver/dataproviders/PyDataProvider.h"
#include "paddle/utils/Util.h"

#include "paddle/testing/TestUtil.h"

using namespace std;     // NOLINT
using namespace paddle;  // NOLINT

void simpleValueCheck(const vector<Argument>& argumentList, bool useGpu);
void simpleSequenceCheck(const vector<Argument>& argumentList, int sample_num);

TEST(PyDataProvider, py_fill_slots) {
  DataConfig config;
  config.set_type("py");
  config.set_async_load_data(false);
  config.set_load_data_module(std::string("pyDataProvider"));
  config.set_load_data_object(std::string("SimpleDataProvider"));
  config.clear_files();
  std::string dataFile = "gserver/tests/pyDataProvider/pyDataProviderList";
  config.set_files(dataFile);
#ifndef PADDLE_WITH_CUDA
  bool useGpu = false;
#else
  bool useGpu = true;
#endif
  unique_ptr<DataProvider> dataProvider(DataProvider::create(config, useGpu));
  DataBatch dataBatch;
  dataProvider->getNextBatchInternal(2, &dataBatch);
  const std::vector<Argument>& argumentList = dataBatch.getStreams();
  // Check size
  EXPECT_EQ(argumentList.size(), 3UL);
  EXPECT_EQ(argumentList[0].value->getWidth(), 3UL);
  EXPECT_EQ(argumentList[0].value->getHeight(), 2UL);
  EXPECT_EQ(argumentList[0].value->getElementCnt(), 6UL);
  EXPECT_EQ(argumentList[1].value->getWidth(), 7UL);
  EXPECT_EQ(argumentList[1].value->getHeight(), 2UL);
  EXPECT_EQ(argumentList[1].value->getElementCnt(), 4UL);
  EXPECT_EQ(argumentList[2].ids->getSize(), 2UL);
  // Check value
  simpleValueCheck(argumentList, useGpu);
  // Check sequenceStartPositions
  simpleSequenceCheck(argumentList, 2);
}

TEST(PyDataProvider, py_fill_nest_slots) {
  DataConfig config;
  config.set_type("py");
  config.set_async_load_data(false);
  config.set_load_data_module(std::string("pyDataProvider"));
  config.set_load_data_object(std::string("SimpleNestDataProvider"));
  config.clear_files();
  std::string dataFile = "gserver/tests/pyDataProvider/pyDataProviderList";
  config.set_files(dataFile);
  EXPECT_EQ(config.IsInitialized(), true);
#ifndef PADDLE_WITH_CUDA
  bool useGpu = false;
#else
  bool useGpu = true;
#endif
  unique_ptr<DataProvider> dataProvider(DataProvider::create(config, useGpu));
  DataBatch dataBatch;
  dataProvider->getNextBatchInternal(2, &dataBatch);
  const std::vector<Argument>& argumentList = dataBatch.getStreams();
  // Check size
  EXPECT_EQ(argumentList.size(), 3UL);
  EXPECT_EQ(argumentList[0].value->getWidth(), 3UL);
  EXPECT_EQ(argumentList[0].value->getHeight(), 4UL);
  EXPECT_EQ(argumentList[0].value->getElementCnt(), 12UL);
  EXPECT_EQ(argumentList[1].value->getWidth(), 7UL);
  EXPECT_EQ(argumentList[1].value->getHeight(), 4UL);
  EXPECT_EQ(argumentList[1].value->getElementCnt(), 8UL);
  EXPECT_EQ(argumentList[2].ids->getSize(), 4UL);
  // Check value
  simpleValueCheck(argumentList, useGpu);
  // Check sequenceStartPositions
  simpleSequenceCheck(argumentList, 4);
  // Check subSequenceStartPositions
  EXPECT_EQ(argumentList[0].subSequenceStartPositions->getSize(), 4UL);
  EXPECT_EQ(argumentList[1].subSequenceStartPositions->getSize(), 3UL);
  EXPECT_EQ(argumentList[2].subSequenceStartPositions->getSize(), 4UL);
  for (size_t i = 0; i < argumentList.size(); i++) {
    EXPECT_EQ(argumentList[i].subSequenceStartPositions->getElement(0), 0);
    EXPECT_EQ(argumentList[i].subSequenceStartPositions->getElement(1), 1);
    if (i != 1) {
      EXPECT_EQ(argumentList[i].subSequenceStartPositions->getElement(2), 2);
      EXPECT_EQ(argumentList[i].subSequenceStartPositions->getElement(3), 4);
    } else {
      EXPECT_EQ(argumentList[i].subSequenceStartPositions->getElement(2), 4);
    }
  }
}

void simpleValueCheck(const vector<Argument>& argumentList, bool useGpu) {
  // Dense
  real* data;
  if (useGpu) {
    MatrixPtr cpuMatrixPtr = Matrix::create(argumentList[0].value->getHeight(),
                                            argumentList[0].value->getWidth(),
                                            0,
                                            0);
    cpuMatrixPtr->copyFrom(*argumentList[0].value);
    data = cpuMatrixPtr->getData();
  } else {
    data = argumentList[0].value->getData();
  }
  for (size_t i = 0; i < argumentList[0].value->getElementCnt(); ++i) {
    EXPECT_EQ(*(data + i), (float)(i % 3 + 1));
  }
  // Sparse without value
  GpuSparseMatrixPtr matGpu;
  CpuSparseMatrixPtr matCpu;
  if (useGpu) {
    matGpu = dynamic_pointer_cast<GpuSparseMatrix>(argumentList[1].value);
    ASSERT_TRUE(matGpu != NULL);
  } else {
    data = argumentList[0].value->getData();
    matCpu = dynamic_pointer_cast<CpuSparseMatrix>(argumentList[1].value);
    ASSERT_TRUE(matCpu != NULL);
  }
  for (size_t i = 0; i < argumentList[1].value->getHeight(); ++i) {
    size_t colNum = useGpu ? matGpu->getColNum(i) : matCpu->getColNum(i);
    EXPECT_EQ(colNum, (size_t)2);
    const int* buf = useGpu ? matGpu->getRowCols(i) : matCpu->getRowCols(i);
    for (size_t j = 0; j < colNum; ++j) {
      EXPECT_EQ((size_t)buf[j], (size_t)(j + 1));
    }
  }
  // Index
  for (size_t j = 0; j < argumentList[2].ids->getSize(); ++j) {
    EXPECT_EQ((size_t)argumentList[2].ids->get(j), 0UL);
  }
}

void simpleSequenceCheck(const vector<Argument>& argumentList, int sample_num) {
  EXPECT_EQ(argumentList[0].sequenceStartPositions->getSize(), 3UL);
  EXPECT_EQ(argumentList[1].sequenceStartPositions->getSize(), 2UL);
  EXPECT_EQ(argumentList[2].sequenceStartPositions->getSize(), 3UL);
  for (size_t i = 0; i < argumentList.size(); i++) {
    EXPECT_EQ(argumentList[i].sequenceStartPositions->getElement(0), 0);
    if (i != 1) {
      EXPECT_EQ(argumentList[i].sequenceStartPositions->getElement(1), 1);
      EXPECT_EQ(argumentList[i].sequenceStartPositions->getElement(2),
                sample_num);
    } else {
      EXPECT_EQ(argumentList[i].sequenceStartPositions->getElement(1),
                sample_num);
    }
  }
}

int main(int argc, char** argv) {
  initMain(argc, argv);
  initPython(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
