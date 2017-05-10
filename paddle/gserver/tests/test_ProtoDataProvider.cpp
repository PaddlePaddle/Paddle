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

#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "paddle/gserver/dataproviders/ProtoDataProvider.h"
#include "paddle/utils/Util.h"

#include "paddle/testing/TestUtil.h"

using namespace std;  // NOLINT

std::vector<string> protoFiles{
    "./test_ProtoDataProvider/data1.bin", "./test_ProtoDataProvider/data2.bin",
};
std::vector<string> protoFilesCompressed{
    "./test_ProtoDataProvider/data1.bin.gz",
    "./test_ProtoDataProvider/data2.bin.gz",
};

const char* kTestDir = "./test_ProtoDataProvider";
const char kProtoFileList[] = "gserver/tests/proto_files.txt";
const char kProtoFileListCompressed[] =
    "gserver/tests/proto_files_compressed.txt";
const int kSpraseMatrixDim = 1024;

using namespace paddle;  // NOLINT

void prepareData(DataBatch* batch,
                 const int* numPerSlotType,
                 bool iid,
                 bool useGpu) {
  batch->clear();
  int64_t size = uniformRandom(100) + 10;
  batch->setSize(size);

  ICpuGpuVectorPtr sequenceStartPositions;
  ICpuGpuVectorPtr subSequenceStartPositions;
  if (!iid) {
    int numSeqs = uniformRandom(10) + 1;
    sequenceStartPositions =
        ICpuGpuVector::create(numSeqs + 1, /* useGpu= */ false);
    int* buf = sequenceStartPositions->getMutableData(false);
    subSequenceStartPositions =
        ICpuGpuVector::create(numSeqs + 1, /* useGpu= */ false);
    int* subBuf = subSequenceStartPositions->getMutableData(false);
    int64_t pos = 0;
    int maxLen = 2 * size / numSeqs;
    for (int i = 0; i < numSeqs; ++i) {
      int len =
          uniformRandom(min<int64_t>(maxLen, size - pos - numSeqs + i)) + 1;
      buf[i] = pos;
      subBuf[i] = pos;
      pos += len;
      VLOG(1) << " len=" << len;
    }
    buf[numSeqs] = size;
    subBuf[numSeqs] = size;
  }

  vector<Argument>& arguments = batch->getStreams();
  for (int i = 0; i < numPerSlotType[SlotDef::VECTOR_DENSE]; ++i) {
    int64_t dim = rand() % 10 + 4;  // NOLINT rand_r
    MatrixPtr mat = Matrix::create(size, dim, /* trans= */ false, false);
    mat->randomizeUniform();
    Argument arg;
    arg.value = mat;
    arg.sequenceStartPositions = sequenceStartPositions;
    arguments.push_back(arg);
  }
  for (int i = 0; i < numPerSlotType[SlotDef::VECTOR_SPARSE_NON_VALUE]; ++i) {
    MatrixPtr mat =
        makeRandomSparseMatrix(size, kSpraseMatrixDim, false, useGpu);
    Argument arg;
    arg.value = mat;
    arg.sequenceStartPositions = sequenceStartPositions;
    arg.subSequenceStartPositions = subSequenceStartPositions;
    arguments.push_back(arg);
  }
  for (int i = 0; i < numPerSlotType[SlotDef::VECTOR_SPARSE_VALUE]; ++i) {
    MatrixPtr mat =
        makeRandomSparseMatrix(size, kSpraseMatrixDim, true, useGpu);
    Argument arg;
    arg.value = mat;
    arg.sequenceStartPositions = sequenceStartPositions;
    arguments.push_back(arg);
  }
  for (int i = 0; i < numPerSlotType[SlotDef::STRING]; ++i) {
    int64_t dim = rand() % 10 + 4;  // NOLINT rand_r
    SVectorPtr vec = std::make_shared<std::vector<std::string>>();
    for (int j = 0; j < size; ++j) {
      vec->push_back(randStr(dim));
    }
    Argument arg;
    arg.strs = vec;
    arg.sequenceStartPositions = sequenceStartPositions;
    arguments.push_back(arg);
  }
  for (int i = 0; i < numPerSlotType[SlotDef::INDEX]; ++i) {
    int64_t dim = rand() % 10 + 4;  // NOLINT rand_r
    IVectorPtr vec = IVector::create(size, /* useGpu= */ false);
    int* buf = vec->getData();
    for (int j = 0; j < size; ++j) {
      buf[j] = uniformRandom(dim);
    }
    Argument arg;
    arg.ids = vec;
    arg.sequenceStartPositions = sequenceStartPositions;
    arguments.push_back(arg);
  }
}

inline int getSlotDim(const Argument& arg) {
  if (arg.value) {
    return arg.value->getWidth();
  } else if (arg.ids) {
    return arg.ids->getMax() + 1;
  } else if (arg.strs) {
    return 1;
  }
  LOG(FATAL) << "Invalid argument";
  return 0;
}

inline SlotDef::SlotType getSlotType(const Argument& arg) {
  if (arg.value) {
    auto& m = *arg.value;
    auto& type = typeid(m);
    if (type == typeid(CpuMatrix) || type == typeid(GpuMatrix)) {
      return SlotDef::VECTOR_DENSE;
    }
    if (type == typeid(CpuSparseMatrix)) {
      auto valueType =
          std::dynamic_pointer_cast<CpuSparseMatrix>(arg.value)->getValueType();
      if (NO_VALUE == valueType) {
        return SlotDef::VECTOR_SPARSE_NON_VALUE;
      } else {
        return SlotDef::VECTOR_SPARSE_VALUE;
      }
    }
    if (type == typeid(GpuSparseMatrix)) {
      auto valueType =
          std::dynamic_pointer_cast<GpuSparseMatrix>(arg.value)->getValueType();
      if (NO_VALUE == valueType) {
        return SlotDef::VECTOR_SPARSE_NON_VALUE;
      } else {
        return SlotDef::VECTOR_SPARSE_VALUE;
      }
    }

    LOG(FATAL) << "Unknown matrix type";
  }
  if (arg.ids) return SlotDef::INDEX;
  if (arg.strs) return SlotDef::STRING;
  LOG(FATAL) << "Invalid argument";
  return SlotDef::VECTOR_DENSE;
}

void getColRow(const Argument& arg,
               int64_t pos,
               bool useGpu,
               int* colNum,
               const int** rowCols,
               const real** rowValues) {
  SlotDef::SlotType type = getSlotType(arg);
  GpuSparseMatrixPtr matGpu;
  CpuSparseMatrixPtr matCpu;
  if (useGpu) {
    matGpu = dynamic_pointer_cast<GpuSparseMatrix>(arg.value);
    ASSERT_TRUE(matGpu != NULL);
  } else {
    matCpu = dynamic_pointer_cast<CpuSparseMatrix>(arg.value);
    ASSERT_TRUE(matCpu != NULL);
  }
  *colNum = useGpu ? matGpu->getColNum(pos) : matCpu->getColNum(pos);
  *rowCols = useGpu ? matGpu->getRowCols(pos) : matCpu->getRowCols(pos);
  if (type == SlotDef::VECTOR_SPARSE_VALUE) {
    *rowValues = useGpu ? matGpu->getRowValues(pos) : matCpu->getRowValues(pos);
  } else {
    *rowValues = NULL;
  }
}

void makeSample(const vector<Argument>& arguments,
                int64_t pos,
                bool isBeginning,
                DataSample* sample,
                bool useGpu) {
  sample->set_is_beginning(isBeginning);
  int slotid = 0;
  for (auto& arg : arguments) {
    SlotDef::SlotType type = getSlotType(arg);
    int64_t dim = getSlotDim(arg);
    switch (type) {
      case SlotDef::VECTOR_DENSE: {
        VectorSlot* vecSlot = sample->add_vector_slots();
        auto values = vecSlot->mutable_values();
        values->Reserve(dim);
        for (int i = 0; i < dim; ++i) {
          values->AddAlreadyReserved(
              static_cast<float>(arg.value->getElement(pos, i)));
        }
        break;
      }
      case SlotDef::INDEX: {
        sample->add_id_slots(arg.ids->get(pos));
        break;
      }
      case SlotDef::VECTOR_SPARSE_NON_VALUE: {
        VectorSlot* vecSlot = sample->add_vector_slots();
        auto ids = vecSlot->mutable_ids();
        int colNum;
        const int* rowCols;
        const real* rowValues;  // nullptr
        getColRow(arg, pos, useGpu, &colNum, &rowCols, &rowValues);
        ids->Reserve(colNum);
        for (int i = 0; i < colNum; ++i) {
          ids->AddAlreadyReserved(rowCols[i]);
        }
        SubseqSlot* subseqSlot = sample->add_subseq_slots();  // subseq
        subseqSlot->set_slot_id(slotid);
        auto lens = subseqSlot->mutable_lens();
        lens->Add(colNum);
        break;
      }
      case SlotDef::VECTOR_SPARSE_VALUE: {
        VectorSlot* vecSlot = sample->add_vector_slots();
        auto values = vecSlot->mutable_values();
        auto ids = vecSlot->mutable_ids();
        int colNum;
        const int* rowCols;
        const real* rowValues;
        getColRow(arg, pos, useGpu, &colNum, &rowCols, &rowValues);
        ids->Reserve(colNum);
        values->Reserve(colNum);
        for (int i = 0; i < colNum; ++i) {
          ids->AddAlreadyReserved(rowCols[i]);
          values->AddAlreadyReserved(rowValues[i]);
        }
        break;
      }
      case SlotDef::VAR_MDIM_DENSE:
      case SlotDef::VAR_MDIM_INDEX: {
        LOG(FATAL) << "Not implemented";
        break;
      }
      case SlotDef::STRING: {
        VectorSlot* vecSlot = sample->add_vector_slots();
        vecSlot->add_strs((*arg.strs)[pos]);
        break;
      }
    }
    slotid++;
  }
}

void writeData(const DataBatch& batch, bool useGpu, bool dataCompression) {
  DataHeader header;
  const vector<Argument>& arguments = batch.getStreams();
  for (auto& argument : arguments) {
    SlotDef* slotDef = header.add_slot_defs();
    slotDef->set_type(getSlotType(argument));
    slotDef->set_dim(getSlotDim(argument));
  }
  VLOG(1) << "header=" << header.DebugString();

  int64_t totalSeqs = batch.getNumSequences();
  int64_t seq = 0;
  ICpuGpuVectorPtr sequenceStartPositions = arguments[0].sequenceStartPositions;
  int64_t numWritten = 0;
  vector<string> curProtoFiles =
      dataCompression ? protoFilesCompressed : protoFiles;
  for (size_t i = 0; i < curProtoFiles.size(); ++i) {
    int64_t numSeqs = totalSeqs * (i + 1) / curProtoFiles.size() -
                      totalSeqs * i / curProtoFiles.size();
    ofstream os(curProtoFiles[i]);
    CHECK(os) << "Fail to open " << curProtoFiles[i];
    unique_ptr<ProtoWriter> writer(new ProtoWriter(&os, dataCompression));
    CHECK(writer->write(header));
    for (int j = 0; j < numSeqs; ++j, ++seq) {
      int64_t begin = seq;
      int64_t end = seq + 1;
      if (sequenceStartPositions) {
        begin = sequenceStartPositions->getElement(seq);
        end = sequenceStartPositions->getElement(seq + 1);
      }
      for (int pos = begin; pos < end; ++pos) {
        DataSample sample;
        makeSample(arguments, pos, pos == begin, &sample, useGpu);
        CHECK(writer->write(sample));
        ++numWritten;
      }
    }

    writer.reset(nullptr);
    os.close();
  }
  CHECK_EQ(arguments[0].getBatchSize(), numWritten);
}

// check that the sample at pos1 in args1 is same as the sample at pos2 in args2
void checkSample(const vector<Argument>& args1,
                 int64_t pos1,
                 const vector<Argument>& args2,
                 int64_t pos2,
                 bool useGpu) {
  EXPECT_EQ(args1.size(), args2.size());
  VLOG(1) << " pos1=" << pos1 << " pos2=" << pos2;

  for (size_t i = 0; i < args1.size(); ++i) {
    auto type = getSlotType(args1[i]);
    int dim = getSlotDim(args1[i]);
    EXPECT_EQ(type, getSlotType(args2[i]));
    if (type == SlotDef::INDEX) {
      EXPECT_GE(dim, getSlotDim(args2[i]));
    } else {
      EXPECT_EQ(dim, getSlotDim(args2[i]));
    }
    switch (type) {
      case SlotDef::VECTOR_DENSE: {
        for (int j = 0; j < dim; ++j) {
          EXPECT_EQ(static_cast<float>(args1[i].value->getElement(pos1, j)),
                    static_cast<float>(args2[i].value->getElement(pos2, j)));
        }
        break;
      }
      case SlotDef::INDEX: {
        EXPECT_EQ(args1[i].ids->get(pos1), args2[i].ids->get(pos2));
        break;
      }
      case SlotDef::VECTOR_SPARSE_NON_VALUE:
      case SlotDef::VECTOR_SPARSE_VALUE: {
        int colNum1, colNum2;
        const int *rowCols1, *rowCols2;
        const real *rowValues1, *rowValues2;
        getColRow(args1[i], pos1, useGpu, &colNum1, &rowCols1, &rowValues1);
        getColRow(args2[i], pos2, useGpu, &colNum2, &rowCols2, &rowValues2);
        EXPECT_EQ(colNum1, colNum2);
        for (int j = 0; j < colNum1; ++j) {
          EXPECT_EQ(rowCols1[j], rowCols2[j]);
          if (type == SlotDef::VECTOR_SPARSE_VALUE) {
            EXPECT_EQ(rowValues1[j], rowValues2[j]);
          }
        }
        break;
      }
      case SlotDef::VAR_MDIM_DENSE:
      case SlotDef::VAR_MDIM_INDEX: {
        LOG(FATAL) << "Not implemented";
        break;
      }
      case SlotDef::STRING: {
        EXPECT_EQ((*args1[i].strs)[pos1], (*args2[i].strs)[pos2]);
        break;
      }
    }
  }
}

void testProtoDataProvider(int* numPerSlotType,
                           bool iid,
                           bool async,
                           bool useGpu,
                           bool dataCompression,
                           int numConstantSlots = 0) {
  mkDir(kTestDir);
  DataBatch data;

  prepareData(&data, numPerSlotType, iid, useGpu);
  writeData(data, useGpu, dataCompression);

  DataConfig config;
  config.set_type("proto");
  config.set_files(dataCompression ? kProtoFileListCompressed : kProtoFileList);
  config.set_async_load_data(async);

  for (int i = 0; i < numConstantSlots; ++i) {
    config.add_constant_slots(i + 11);
    MatrixPtr w = Matrix::create(data.getSize(),
                                 1,
                                 /* trans= */ false,
                                 /* useGpu= */ false);
    w->assign(config.constant_slots(i));
    data.appendData(w);
  }

  unique_ptr<DataProvider> dataProvider(DataProvider::create(config, useGpu));
  dataProvider->setSkipShuffle();

  EXPECT_EQ(data.getSize(), dataProvider->getSize());

  int64_t batchSize = 10;
  DataBatch batch;

  size_t seq1 = 0;
  vector<Argument>& args1 = data.getStreams();
  ICpuGpuVectorPtr sequenceStartPositions1 = args1[0].sequenceStartPositions;

  dataProvider->reset();

  while (dataProvider->getNextBatch(batchSize, &batch) > 0) {
    CHECK_EQ(data.getNumStreams(), batch.getNumStreams());
    vector<Argument>& args2 = batch.getStreams();
    ICpuGpuVectorPtr sequenceStartPositions2 = args2[0].sequenceStartPositions;
    for (auto& arg : args2) {
      EXPECT_EQ(iid, !arg.sequenceStartPositions);
    }
    size_t numSeqs = batch.getNumSequences();
    VLOG(1) << "numSeqs=" << numSeqs;
    for (size_t seq2 = 0; seq2 < numSeqs; ++seq1, ++seq2) {
      int64_t begin1 = seq1;
      int64_t end1 = seq1 + 1;
      if (sequenceStartPositions1) {
        begin1 = sequenceStartPositions1->getElement(seq1);
        end1 = sequenceStartPositions1->getElement(seq1 + 1);
        EXPECT_LT(seq1, sequenceStartPositions1->getSize() - 1);
      }

      int64_t begin2 = seq2;
      int64_t end2 = seq2 + 1;
      if (sequenceStartPositions2) {
        begin2 = sequenceStartPositions2->getElement(seq2);
        end2 = sequenceStartPositions2->getElement(seq2 + 1);
      }
      VLOG(1) << " begin1=" << begin1 << " end1=" << end1
              << " begin2=" << begin2 << " end2=" << end2;
      EXPECT_EQ(end1 - begin1, end2 - begin2);
      for (int i = 0; i < end1 - begin1; ++i) {
        checkSample(args1, begin1 + i, args2, begin2 + i, useGpu);
      }
    }
  }

  EXPECT_EQ(seq1, (size_t)data.getNumSequences());
  rmDir(kTestDir);
}

TEST(ProtoDataProvider, test) {
  int numSlotsArray[] = {0, 3};
  int numTwoArray[] = {0, 1};
  int numSlotsArraySize = sizeof(numSlotsArray) / sizeof(numSlotsArray[0]);
  const int numSlot = 5;
  int combination[numSlot] = {0};
  int k = numSlot - 1;
  while (k >= 0) {
    int numDenseVecSlots = numSlotsArray[combination[0]];
    int numSparseNonValueVecSlots = numSlotsArray[combination[1]];
    int numSparseValueVectorSlots = numSlotsArray[combination[2]];
    int numStrSlots = numSlotsArray[combination[3]];
    int numIdSlots = numSlotsArray[combination[4]];
    // while loop : traverse all cases
    k = numSlot - 1;
    while (k >= 0) {
      if (combination[k] < (numSlotsArraySize - 1)) {
        ++combination[k];
        break;
      } else {
        combination[k] = 0;
        --k;
      }
    }
    if (numDenseVecSlots + numSparseNonValueVecSlots +
            numSparseValueVectorSlots + numStrSlots + numIdSlots <
        1)
      continue;
    for (int iid : numTwoArray) {
      for (int async : numTwoArray) {
        for (int useGpu : numTwoArray) {
          for (int dataCompression : numTwoArray) {
            if (async && useGpu) {
              // Currently in async mode, useGpu is not supported
              continue;
            }
#ifdef PADDLE_ONLY_CPU
            if (useGpu) {
              continue;
            }
#endif
            LOG(INFO) << " numDenseVecSlots=" << numDenseVecSlots
                      << " numSparseNonValueVecSlots="
                      << numSparseNonValueVecSlots
                      << " numSparseValueVectorSlots="
                      << numSparseValueVectorSlots
                      << " numStrSlots=" << numStrSlots
                      << " numIdSlots=" << numIdSlots << " iid=" << iid
                      << " async=" << async << " useGpu=" << useGpu
                      << " dataCompression=" << dataCompression;
            int numPerSlotType[SlotDef::SlotType_ARRAYSIZE] = {0};
            numPerSlotType[SlotDef::VECTOR_DENSE] = numDenseVecSlots;
            numPerSlotType[SlotDef::VECTOR_SPARSE_NON_VALUE] =
                numSparseNonValueVecSlots;
            numPerSlotType[SlotDef::VECTOR_SPARSE_VALUE] =
                numSparseValueVectorSlots;
            numPerSlotType[SlotDef::INDEX] = numIdSlots;
            numPerSlotType[SlotDef::STRING] = numStrSlots;
            testProtoDataProvider(
                numPerSlotType, iid, async, useGpu, dataCompression);
          }  // end for (int dataCompression : numTwoArray)
        }    // end for (int useGpu : numTwoArray)
      }      // end for (int async : numTwoArray)
    }        // end for (int iid : numTwoArray)
  }          // end for (while, traverse all slots)
}

TEST(ProtoDataProvider, constant_slots) {
  int numSlotsArray[] = {0, 3};
  int numTwoArray[] = {0, 1};
  for (int numDenseVecSlots : numSlotsArray) {
    for (int numSparseNonValueVecSlots : numSlotsArray) {
      if (numDenseVecSlots + numSparseNonValueVecSlots < 1) continue;
      for (int numConstantSlots : {1, 2}) {
        for (int useGpu : numTwoArray) {
          for (int dataCompression : numTwoArray) {
#ifdef PADDLE_ONLY_CPU
            if (useGpu) {
              continue;
            }
#endif
            LOG(INFO) << " numDenseVecSlots=" << numDenseVecSlots
                      << " numSparseNonValueVecSlots="
                      << numSparseNonValueVecSlots
                      << " numConstantSlogs=" << numConstantSlots
                      << " useGpu=" << useGpu
                      << " dataCompression=" << dataCompression;
            int numPerSlotType[SlotDef::SlotType_ARRAYSIZE] = {0};
            numPerSlotType[SlotDef::VECTOR_DENSE] = numDenseVecSlots;
            numPerSlotType[SlotDef::VECTOR_SPARSE_NON_VALUE] =
                numSparseNonValueVecSlots;
            numPerSlotType[SlotDef::VECTOR_SPARSE_VALUE] = 1;
            numPerSlotType[SlotDef::INDEX] = 1;
            testProtoDataProvider(numPerSlotType,
                                  /* iid= */ true,
                                  /* async= */ false,
                                  useGpu,
                                  dataCompression,
                                  numConstantSlots);
          }  // end for (int dataCompression : numTwoArray)
        }    // end for (int useGpu : numTwoArray)
      }      // end for (int numConstantSlots : {1, 2})
    }        // end for (int numSparseNonValueVecSlots : numSlotsArray)
  }          // end for (int numDenseVecSlots : numSlotsArray)
}

void checkSampleSequence(const vector<Argument>& args1,
                         const vector<Argument>& args2,
                         int64_t offset,
                         int64_t numSeqs,
                         bool useGpu) {
  // check slot num are equal
  EXPECT_EQ(args1.size(), args2.size());
  for (size_t i = 0; i < args1.size(); i++) {
    auto type = getSlotType(args1[i]);
    // check for args2: sequenceStartPositions vs numSeqs
    // (1) size
    EXPECT_EQ(args2[i].sequenceStartPositions->getSize(), (size_t)numSeqs + 1);
    // (2) content
    auto checkArgContent = [&](const Argument& args, int numSeqs) {
      for (int j = 0; j <= numSeqs; j++) {
        int start_pos = args.sequenceStartPositions->getElement(j);
        EXPECT_EQ(start_pos, j);
      }
    };
    switch (type) {
      case SlotDef::INDEX: {
        // args1: for label
        checkArgContent(args2[i], numSeqs);
        // check for args2: ids are equal to args1[offset]
        // (1) size
        EXPECT_EQ(args2[i].ids->getSize(), (size_t)numSeqs);
        // (2) content
        for (int j = 0; j < numSeqs; j++) {
          EXPECT_EQ(args2[i].ids->get(j), args1[i].ids->get(offset + j));
        }
        break;
      }
      case SlotDef::VECTOR_SPARSE_NON_VALUE: {
        // args1: for sparse_non_value
        // args2 should put sparse indexes in ids
        int colNum1;
        const int* rowCols1;
        const real* rowValues1;  // nullptr
        int totalLength = 0;
        for (int j = 0; j < numSeqs; j++) {
          getColRow(
              args1[i], offset + j, useGpu, &colNum1, &rowCols1, &rowValues1);
          // (1) lengths
          EXPECT_EQ(totalLength,
                    args2[i].sequenceStartPositions->getElement(j));
          EXPECT_EQ(totalLength,
                    args2[i].subSequenceStartPositions->getElement(j));
          // (2) content
          for (int k = 0; k < colNum1; k++) {
            EXPECT_EQ(rowCols1[k], args2[i].ids->get(totalLength + k));
          }
          totalLength += colNum1;
          if (colNum1 == 0) {
            // special case here: we will put a "-1" into ids when column num is
            // zero. see ProtoSequenceDataProvider::getNextBatchInternal.
            EXPECT_EQ(-1, args2[i].ids->get(totalLength));
            totalLength++;
          }
        }
        EXPECT_EQ(totalLength,
                  args2[i].sequenceStartPositions->getElement(numSeqs));
        EXPECT_EQ(totalLength,
                  args2[i].subSequenceStartPositions->getElement(numSeqs));
        break;
      }
      case SlotDef::VECTOR_DENSE: {
        // args1: for dense vector
        checkArgContent(args2[i], numSeqs);
        // check for args2: values are equal to args1[offset]
        // (1) size
        EXPECT_EQ(args2[i].value->getHeight(), (size_t)numSeqs);
        EXPECT_EQ(args2[i].value->getWidth(), (size_t)getSlotDim(args1[i]));
        // (2) content
        for (int j = 0; j < numSeqs; j++) {
          for (size_t k = 0; k < args2[i].value->getWidth(); k++) {
            EXPECT_EQ(
                static_cast<float>(args1[i].value->getElement(j + offset, k)),
                static_cast<float>(args2[i].value->getElement(j, k)));
          }
        }
        break;
      }
      default: { EXPECT_EQ(true, false) << "should not reach here"; }
    }
  }
}

void testProtoSequenceDataProvider(int* numPerSlotType,
                                   bool async,
                                   bool useGpu) {
  mkDir(kTestDir);
  DataBatch data;

  prepareData(&data,
              numPerSlotType,
              /* iid */ true,
              useGpu);
  writeData(data, useGpu, /* dataCompression */ false);

  DataConfig config;
  config.set_type("proto_sequence");
  config.set_files(kProtoFileList);
  config.set_async_load_data(async);

  unique_ptr<DataProvider> dataProvider(DataProvider::create(config, useGpu));
  dataProvider->setSkipShuffle();

  EXPECT_EQ(data.getSize(), dataProvider->getSize());

  int64_t batchSize = 10;
  DataBatch batch;

  vector<Argument>& args1 = data.getStreams();
  ICpuGpuVectorPtr sequenceStartPositions1 = args1[0].sequenceStartPositions;

  dataProvider->reset();

  size_t args1Offset = 0;
  while (dataProvider->getNextBatch(batchSize, &batch) > 0) {
    CHECK_EQ(data.getNumStreams(), batch.getNumStreams());
    vector<Argument>& args2 = batch.getStreams();
    ICpuGpuVectorPtr sequenceStartPositions2 = args2[0].sequenceStartPositions;
    for (auto& arg : args1) {
      // args1 should not has sequence
      EXPECT_EQ(true, !arg.sequenceStartPositions);
    }
    for (auto& arg : args2) {
      // args2 should has sequence
      EXPECT_NE(true, !arg.sequenceStartPositions);
    }
    size_t numSeqs = batch.getNumSequences();
    checkSampleSequence(args1, args2, args1Offset, numSeqs, useGpu);
    args1Offset += numSeqs;
  }

  EXPECT_EQ(args1Offset, (size_t)data.getNumSequences());
  rmDir(kTestDir);
}

TEST(ProtoSequenceDataProvider, test) {
  int numSlotsArray[] = {0, 3};
  int numTwoArray[] = {0, 1};
  for (int numSparseNonValueVecSlots : numSlotsArray) {
    for (int numIdSlots : numSlotsArray) {
      for (int numDenseVecSlots : numSlotsArray) {
        if (numDenseVecSlots + numSparseNonValueVecSlots + numIdSlots < 1)
          continue;
        for (int async : numTwoArray) {
          for (int useGpu : numTwoArray) {
            if (async && useGpu) {
              // Currently in async mode, useGpu is not supported
              continue;
            }
#ifdef PADDLE_ONLY_CPU
            if (useGpu) {
              continue;
            }
#endif
            LOG(INFO) << " numDenseVecSlots=" << numDenseVecSlots
                      << " numSparseNonValueVecSlots="
                      << numSparseNonValueVecSlots
                      << " numIdSlots=" << numIdSlots << " async=" << async
                      << " useGpu=" << useGpu;
            int numPerSlotType[SlotDef::SlotType_ARRAYSIZE] = {0};
            numPerSlotType[SlotDef::VECTOR_DENSE] = numDenseVecSlots;
            numPerSlotType[SlotDef::VECTOR_SPARSE_NON_VALUE] =
                numSparseNonValueVecSlots;
            numPerSlotType[SlotDef::INDEX] = numIdSlots;
            testProtoSequenceDataProvider(numPerSlotType, async, useGpu);
          }  // end for (int useGpu : numTwoArray)
        }    // end for (int async : numTwoArray)
      }      // end for (int numDenseVecSlots : numSlotsArray)
    }        // end for (int numIdSlots : numSlotsArray)
  }          // end for (int numSparseNonValueVecSlots : numSlotsArray)
}
