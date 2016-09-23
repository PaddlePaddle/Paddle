/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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
#include <random>
#include <fstream>

#include <gtest/gtest.h>
#include "DataFormat.pb.h"

#include "paddle/utils/Util.h"
#include "paddle/gserver/dataproviders/DataProvider.h"
#include "paddle/gserver/dataproviders/ProtoReader.h"

using namespace paddle;  //NOLINT


std::vector<std::string> protoFiles{
        "./test_proto/data1.bin",
        "./test_proto/data2.bin",
};
std::vector<std::string> protoFilesCompressed{
        "./test_proto/data1.bin.gz",
        "./test_proto/data2.bin.gz",
};


const char* kTestDir = "./test_proto";
const char kProtoFileList[] = "./test_proto/proto_files.txt";
const char kProtoFileListCompressed[] =
        "./test_proto/proto_files_compressed.txt";

const int slotDim = 100;
const int batchSize = 1024;
const int seqLength = 10;
const int subSeqLength = 5;
const paddle::real epsilon = 1e-5;

std::random_device dev;
std::mt19937_64 rng(dev());
std::uniform_int_distribution<int> distId(0, slotDim - 1);
std::uniform_real_distribution<float> distValue(-1, 1);

void writeData(DataHeader2& header, std::vector<DataSample2>& samples,
               bool dataCompression) {
  std::vector<std::string> curProtoFiles =
    dataCompression ? protoFilesCompressed : protoFiles;
  std::ofstream out(dataCompression ?
                    kProtoFileListCompressed : kProtoFileList);
  out << curProtoFiles[0] + "\n";
  out << curProtoFiles[1];
  out.close();

  for (size_t i = 0; i < curProtoFiles.size(); i++) {
    std::cout << curProtoFiles[i] << std::endl;
    std::ofstream os(curProtoFiles[i]);
    CHECK(os) << "Fail to open " << curProtoFiles[i];
    std::unique_ptr<ProtoWriter> writer(new ProtoWriter(&os, dataCompression));
    CHECK(writer->write(header));
    for (size_t j = 0; j < samples.size(); ++j) {
      CHECK(writer->write(samples[j]));
    }
  }
}

DataProvider* createProvider(bool useGpu, bool dataCompression, bool async) {
  DataConfig config;
  config.set_type("proto2");
  config.set_files(dataCompression ? kProtoFileListCompressed : kProtoFileList);
  config.set_async_load_data(async);
  return DataProvider::create(config, useGpu);
}


void getColRow(MatrixPtr mat, SlotDef::SlotType type, bool useGpu,
               int64_t pos, int* colNum, const int** rowCols,
               const real** rowValues) {
  GpuSparseMatrixPtr matGpu;
  CpuSparseMatrixPtr matCpu;
  if (useGpu) {
    matGpu = std::dynamic_pointer_cast<GpuSparseMatrix>(mat);
  } else {
    matCpu = std::dynamic_pointer_cast<CpuSparseMatrix>(mat);
  }
  *colNum = useGpu ? matGpu->getColNum(pos) : matCpu->getColNum(pos);
  *rowCols = useGpu ? matGpu->getRowCols(pos) : matCpu->getRowCols(pos);
  if (type == SlotDef::VECTOR_SPARSE_VALUE) {
    *rowValues = useGpu ? matGpu->getRowValues(pos) : matCpu->getRowValues(pos);
  } else {
    *rowValues = NULL;
  }
}


TEST(ProtoDataProvider2, testDenseSlot) {
  int numTwoArray[] = {0, 1};
  for (int useGpu : numTwoArray) {
    for (int dataCompression : numTwoArray) {
      for (int async : numTwoArray) {
        if (async && useGpu) {
          continue;
        }
#ifdef PADDLE_ONLY_CPU
        if (useGpu) {
          continue;
        }
#endif
        DataHeader2 header;
        auto def = header.add_slot_defs();
        def->set_dim(slotDim);
        def->set_type(SlotDef::VECTOR_DENSE);
        header.add_seq_type(DataHeader2::NON_SEQ);

        std::vector<float> genValue;
        std::vector<DataSample2> samples(batchSize);
        for (int i = 0; i < batchSize; ++i) {
          SlotSample *slotSample = samples[i].add_slots_data();
          slotSample->set_slot_id(0);
          VectorSlot *vec = slotSample->add_vector_slots();
          vec->add_dims(slotDim);
          auto valueBuffer = vec->mutable_values();
          valueBuffer->Reserve(slotDim);
          for (int j = 0; j < slotDim; ++j) {
            float temp = distValue(rng);
            valueBuffer->AddAlreadyReserved(temp);
            genValue.push_back(temp);
          }
        }
        mkDir(kTestDir);
        writeData(header, samples, dataCompression);
        DataProvider *provider = createProvider(useGpu, dataCompression, async);
        provider->setSkipShuffle();
        provider->reset();

        int64_t size = 100;
        DataBatch batchOut;
        int actBatchSize = 0;
        int testRowId = 0;
        do {
          actBatchSize = provider->getNextBatch(size, &batchOut);
          if (actBatchSize > 0) {
            const Argument &arg = batchOut.getStream(0);
            MatrixPtr matCpu = std::make_shared<CpuMatrix>(actBatchSize,
                                                           slotDim);
            matCpu->copyFrom(*(arg.value));
            for (int i = 0; i < actBatchSize; i++) {
              real *row = matCpu->getData() + i * slotDim;
              for (int j = 0; j < slotDim; j++) {
                ASSERT_EQ(genValue[((testRowId + i) * slotDim + j) %
                                   (batchSize * slotDim)], row[j]);
              }
            }
            testRowId += actBatchSize;
          }
        } while (actBatchSize != 0);
        rmDir(kTestDir);
      }
    }
  }
}

TEST(ProtoDataProvider2, testIndexSlot) {
  int numTwoArray[] = {0, 1};
  for (int useGpu : numTwoArray) {
    for (int dataCompression : numTwoArray) {
      for (int async : numTwoArray) {
        if (async && useGpu) {
          continue;
        }
#ifdef PADDLE_ONLY_CPU
        if (useGpu) {
          continue;
        }
#endif
        DataHeader2 header;
        auto def = header.add_slot_defs();
        def->set_dim(slotDim);
        def->set_type(SlotDef::INDEX);
        header.add_seq_type(DataHeader2::NON_SEQ);

        std::vector<int> genId;
        std::vector<DataSample2> samples(batchSize);
        for (int i = 0; i < batchSize; ++i) {
          SlotSample *slotSample = samples[i].add_slots_data();
          slotSample->set_slot_id(0);
          VectorSlot *vec = slotSample->add_vector_slots();
          vec->add_dims(slotDim);
          int temp = distId(rng);
          vec->add_ids(temp);
          genId.push_back(temp);
        }

        mkDir("./test_proto");
        writeData(header, samples, dataCompression);
        DataProvider *provider = createProvider(useGpu, dataCompression, async);
        provider->setSkipShuffle();
        provider->reset();

        int64_t size = 100;
        DataBatch batchOut;
        int actBatchSize = 0;
        int testRowId = 0;
        do {
          actBatchSize = provider->getNextBatch(size, &batchOut);
          if (actBatchSize > 0) {
            const Argument &arg = batchOut.getStream(0);
            IVectorPtr temp = IVector::create(actBatchSize, false);
            temp->copyFrom(*arg.ids);
            for (int i = 0; i < actBatchSize; i++) {
              ASSERT_EQ(genId[(testRowId + i) % batchSize], temp->getData()[i]);
            }
            testRowId += actBatchSize;
          }
        } while (actBatchSize != 0);
        rmDir("./test_proto");
      }
    }
  }
}

TEST(ProtoDataProvider2, testSparseNonValue) {
  int numTwoArray[] = {0, 1};
  for (int useGpu : numTwoArray) {
    for (int dataCompression : numTwoArray) {
      for (int async : numTwoArray) {
        if (async && useGpu) {
          continue;
        }
#ifdef PADDLE_ONLY_CPU
        if (useGpu) {
          continue;
        }
#endif
        DataHeader2 header;
        auto def = header.add_slot_defs();
        def->set_dim(slotDim);
        def->set_type(SlotDef::VECTOR_SPARSE_NON_VALUE);
        header.add_seq_type(DataHeader2::NON_SEQ);

        std::vector<std::set<int>> genIds;
        std::vector<DataSample2> samples(batchSize);
        for (int i = 0; i < batchSize; ++i) {
          SlotSample *slotSample = samples[i].add_slots_data();
          slotSample->set_slot_id(0);
          VectorSlot *vec = slotSample->add_vector_slots();
          vec->add_dims(slotDim);
          std::vector<int> genIdVec;
          for (int k = 0; k < slotDim; ++k) {
            genIdVec.push_back(distId(rng));
          }
          std::set<int> genIdSet(genIdVec.begin(), genIdVec.end());
          genIds.push_back(genIdSet);
          auto valueBufferIds = vec->mutable_ids();
          valueBufferIds->Reserve(genIdSet.size());
          for (auto id : genIdSet) {
            valueBufferIds->AddAlreadyReserved(id);
          }
        }

        mkDir(kTestDir);
        writeData(header, samples, dataCompression);
        DataProvider *provider = createProvider(useGpu, dataCompression, async);
        provider->setSkipShuffle();
        provider->reset();

        int64_t size = 100;
        DataBatch batchOut;
        int actBatchSize = 0;
        int testRowId = 0;
        do {
          actBatchSize = provider->getNextBatch(size, &batchOut);
          if (actBatchSize > 0) {
            const Argument &arg = batchOut.getStream(0);
            MatrixPtr mat = nullptr;
            if (useGpu) {
              MatrixPtr cpuMat(new CpuSparseMatrix(actBatchSize, slotDim,
                                                   arg.value->getElementCnt(),
                                                   NO_VALUE));
              cpuMat->copyFrom(*arg.value, HPPL_STREAM_1);
              hl_stream_synchronize(HPPL_STREAM_1);
              mat = cpuMat;
            } else {
              mat = std::dynamic_pointer_cast<CpuSparseMatrix>(arg.value);
            }

            for (int i = 0; i < actBatchSize; i++) {
              int colNum;
              const int *rowCols;
              const real *rowValues;
              getColRow(mat, SlotDef::VECTOR_SPARSE_NON_VALUE, false, i,
                        &colNum, &rowCols, &rowValues);
              int genPos = (i + testRowId) % batchSize;
              ASSERT_EQ(genIds[genPos].size(), colNum);
              std::vector<int> temp(genIds[genPos].size());
              std::copy(genIds[genPos].begin(),
                        genIds[genPos].end(), temp.begin());
              for (int j = 0; j < colNum; ++j) {
                ASSERT_EQ(temp[j], rowCols[j]);
              }
            }
            testRowId += actBatchSize;
          }
        } while (actBatchSize != 0);
        rmDir(kTestDir);
      }
    }
  }
}

TEST(ProtoDataProvider2, testSparseValue) {
  int numTwoArray[] = {0, 1};
  for (int useGpu : numTwoArray) {
    for (int dataCompression : numTwoArray) {
      for (int async : numTwoArray) {
        if (async && useGpu) {
          continue;
        }
#ifdef PADDLE_ONLY_CPU
        if (useGpu) {
          continue;
        }
#endif
        DataHeader2 header;
        auto def = header.add_slot_defs();
        def->set_dim(slotDim);
        def->set_type(SlotDef::VECTOR_SPARSE_VALUE);
        header.add_seq_type(DataHeader2::NON_SEQ);

        std::vector<std::set<int>> genIds;
        std::vector<std::vector<float>> genValues;
        std::vector<DataSample2> samples(batchSize);
        for (int i = 0; i < batchSize; ++i) {
          SlotSample *slotSample = samples[i].add_slots_data();
          slotSample->set_slot_id(0);
          VectorSlot *vec = slotSample->add_vector_slots();
          vec->add_dims(slotDim);
          std::vector<int> genIdVec;
          std::vector<float> genValueVec;
          for (int k = 0; k < slotDim; ++k) {
            genIdVec.push_back(distId(rng));
          }
          std::set<int> genIdSet(genIdVec.begin(), genIdVec.end());
          genIds.push_back(genIdSet);
          auto valueBufferIds = vec->mutable_ids();
          auto valueBufferValues = vec->mutable_values();
          valueBufferIds->Reserve(genIdSet.size());
          valueBufferValues->Reserve(genIdSet.size());
          for (auto id : genIdSet) {
            valueBufferIds->AddAlreadyReserved(id);
            float temp = distValue(rng);
            valueBufferValues->AddAlreadyReserved(temp);
            genValueVec.push_back(temp);
          }
          genValues.push_back(genValueVec);
        }

        mkDir(kTestDir);
        writeData(header, samples, dataCompression);
        DataProvider *provider = createProvider(useGpu, dataCompression, async);
        provider->setSkipShuffle();
        provider->reset();

        int64_t size = 100;
        DataBatch batchOut;
        int actBatchSize = 0;
        int testRowId = 0;
        do {
          actBatchSize = provider->getNextBatch(size, &batchOut);
          if (actBatchSize > 0) {
            const Argument &arg = batchOut.getStream(0);
            MatrixPtr mat = nullptr;
            if (useGpu) {
              MatrixPtr cpuMat(new CpuSparseMatrix(actBatchSize, slotDim,
                                                   arg.value->getElementCnt(),
                                                   FLOAT_VALUE));
              cpuMat->copyFrom(*arg.value, HPPL_STREAM_1);
              hl_stream_synchronize(HPPL_STREAM_1);
              mat = cpuMat;
            } else {
              mat = std::dynamic_pointer_cast<CpuSparseMatrix>(arg.value);
            }
            for (int i = 0; i < actBatchSize; i++) {
              int colNum;
              const int *rowCols;
              const real *rowValues;
              getColRow(mat, SlotDef::VECTOR_SPARSE_VALUE, false, i, &colNum,
                        &rowCols, &rowValues);
              int genPos = (i + testRowId) % batchSize;
              ASSERT_EQ(genIds[genPos].size(), colNum);
              std::vector<int> tempId(genIds[genPos].size());
              std::copy(genIds[genPos].begin(), genIds[genPos].end(),
                        tempId.begin());
              for (int j = 0; j < colNum; ++j) {
                ASSERT_EQ(tempId[j], rowCols[j]);
                ASSERT_EQ(genValues[genPos][j], rowValues[j]);
              }
            }
            testRowId += actBatchSize;
          }
        } while (actBatchSize != 0);
        rmDir(kTestDir);
      }
    }
  }
}

TEST(ProtoDataProvider2, testSequenceSlot) {
  int numTwoArray[] = {0, 1};
  for (int useGpu : numTwoArray) {
    for (int dataCompression : numTwoArray) {
      for (int async : numTwoArray) {
        if (async && useGpu) {
          continue;
        }
#ifdef PADDLE_ONLY_CPU
        if (useGpu) {
          continue;
        }
#endif
        DataHeader2 header;
        auto def = header.add_slot_defs();
        def->set_dim(slotDim);
        def->set_type(SlotDef::VECTOR_DENSE);
        header.add_seq_type(DataHeader2::SEQ);

        std::vector<float> genValue;
        std::vector<DataSample2> samples(batchSize);


        for (int i = 0; i < batchSize; ++i) {
          SlotSample *slotSample = samples[i].add_slots_data();
          slotSample->set_slot_id(0);
          for (int j = 0; j < seqLength; ++j) {
            VectorSlot *vec = slotSample->add_vector_slots();
            auto valueBuffer = vec->mutable_values();
            valueBuffer->Reserve(slotDim);
            for (int k = 0; k < slotDim; ++k) {
              float temp = distValue(rng);
              valueBuffer->AddAlreadyReserved(temp);
              genValue.push_back(temp);
            }
          }
        }

        mkDir(kTestDir);
        writeData(header, samples, dataCompression);
        DataProvider *provider = createProvider(useGpu, dataCompression, async);
        provider->setSkipShuffle();
        provider->reset();

        int64_t size = 100;
        DataBatch batchOut;
        int actBatchSize = 0;
        int testRowId = 0;
        do {
          actBatchSize = provider->getNextBatch(size, &batchOut);
          if (actBatchSize > 0) {
            const Argument &arg = batchOut.getStream(0);
            int *seqStartPos;
            if (useGpu) {
              ICpuGpuVectorPtr cpuSequenceStartPositions =
                  ICpuGpuVector::create(actBatchSize, false);
              cpuSequenceStartPositions->copyFrom(*arg.sequenceStartPositions,
                                                  HPPL_STREAM_1);
              seqStartPos = cpuSequenceStartPositions->getMutableData(false);
            } else {
              seqStartPos = arg.sequenceStartPositions->getMutableData(false);
            }
            MatrixPtr matCpu = std::make_shared<CpuMatrix>(actBatchSize,
                                                           slotDim * seqLength);
            matCpu->copyFrom(*(arg.value));
            for (int i = 0; i < actBatchSize; ++i) {
              ASSERT_EQ(seqLength * i, seqStartPos[i]);
              for (int j = 0; j < seqLength; ++j) {
                real *row = matCpu->getData() + (seqLength * i + j) * slotDim;
                for (int k = 0; k < slotDim; ++k) {
                  ASSERT_NEAR(genValue[((testRowId + i) * seqLength * slotDim +
                                        j * slotDim + k) %
                                       (batchSize * seqLength * slotDim)],
                              row[k],
                              epsilon);
                }
              }
            }
            if (actBatchSize != 0) {
              ASSERT_EQ(seqLength * actBatchSize, seqStartPos[actBatchSize]);
            }
            testRowId += actBatchSize;
          }
        } while (actBatchSize != 0);
        rmDir(kTestDir);
      }
    }
  }
}

TEST(ProtoDataProvider2, testSubSequenceSlot) {
  int numTwoArray[] = {0, 1};
  for (int useGpu : numTwoArray) {
    for (int dataCompression : numTwoArray) {
      for (int async : numTwoArray) {
        if (async && useGpu) {
          continue;
        }
#ifdef PADDLE_ONLY_CPU
        if (useGpu) {
          continue;
        }
#endif
        DataHeader2 header;
        auto def = header.add_slot_defs();
        def->set_dim(slotDim);
        def->set_type(SlotDef::VECTOR_DENSE);
        header.add_seq_type(DataHeader2::SUB_SEQ);

        std::vector<float> genValue;
        std::vector<DataSample2> samples(batchSize);

        for (int i = 0; i < batchSize; ++i) {
          SlotSample *slotSample = samples[i].add_slots_data();
          slotSample->set_slot_id(0);
          for (int j = 0; j < seqLength; ++j) {
            VectorSlot *vec = slotSample->add_vector_slots();
            auto valueBuffer = vec->mutable_values();
            valueBuffer->Reserve(slotDim);
            for (int k = 0; k < slotDim; ++k) {
              float temp = distValue(rng);
              valueBuffer->AddAlreadyReserved(temp);
              genValue.push_back(temp);
            }
            if (j % subSeqLength == 0) {
              slotSample->add_subseq_start_positions(j);
            }
          }
        }

        mkDir(kTestDir);
        writeData(header, samples, dataCompression);
        DataProvider *provider = createProvider(useGpu, dataCompression, async);
        provider->setSkipShuffle();
        provider->reset();

        int64_t size = 100;
        DataBatch batchOut;
        int actBatchSize = 0;
        int testRowId = 0;
        do {
          actBatchSize = provider->getNextBatch(size, &batchOut);
          if (actBatchSize > 0) {
            const Argument &arg = batchOut.getStream(0);
            int *seqStartPos;
            int *subSeqStartPos;
            if (useGpu) {
              ICpuGpuVectorPtr cpuSequenceStartPositions =
                  ICpuGpuVector::create(actBatchSize, false);
              cpuSequenceStartPositions->copyFrom(*arg.sequenceStartPositions,
                                                  HPPL_STREAM_1);
              seqStartPos = cpuSequenceStartPositions->getMutableData(false);

              ICpuGpuVectorPtr cpuSubSequenceStartPositions =
                  ICpuGpuVector::create(2 * actBatchSize, false);
              cpuSubSequenceStartPositions->copyFrom(
                  *arg.subSequenceStartPositions,
                  HPPL_STREAM_1);
              subSeqStartPos = cpuSubSequenceStartPositions->
                  getMutableData(false);
            } else {
              seqStartPos = arg.sequenceStartPositions->getMutableData(false);
              subSeqStartPos = arg.subSequenceStartPositions->
                  getMutableData(false);
            }
            MatrixPtr matCpu = std::make_shared<CpuMatrix>(actBatchSize,
                                                           slotDim * seqLength);
            matCpu->copyFrom(*(arg.value));
            for (int i = 0; i < actBatchSize; ++i) {
              ASSERT_EQ(seqLength * i, seqStartPos[i]);
              ASSERT_EQ(subSeqLength * 2 * i, subSeqStartPos[2 * i]);
              ASSERT_EQ(subSeqLength * (2 * i + 1), subSeqStartPos[2 * i + 1]);
              for (int j = 0; j < seqLength; ++j) {
                real *row =
                    matCpu->getData() + (seqLength * i + j) * slotDim;
                for (int k = 0; k < slotDim; ++k) {
                  ASSERT_NEAR(genValue[((testRowId + i) * seqLength * slotDim +
                                        j * slotDim + k) %
                                       (batchSize * seqLength * slotDim)],
                              row[k],
                              epsilon);
                }
              }
            }
            if (actBatchSize != 0) {
              ASSERT_EQ(seqLength * actBatchSize, seqStartPos[actBatchSize]);
              ASSERT_EQ(seqLength * actBatchSize,
                        subSeqStartPos[actBatchSize * 2]);
            }
            testRowId += actBatchSize;
          }
        } while (actBatchSize != 0);
        rmDir(kTestDir);
      }
    }
  }
}

TEST(ProtoDataProvider2, testMultiPass) {
  int numTwoArray[] = {0, 1};
  for (int useGpu : numTwoArray) {
    for (int dataCompression : numTwoArray) {
      for (int async : numTwoArray) {
        if (async && useGpu) {
          continue;
        }
#ifdef PADDLE_ONLY_CPU
        if (useGpu) {
          continue;
        }
#endif
        DataHeader2 header;
        auto def = header.add_slot_defs();
        def->set_dim(slotDim);
        def->set_type(SlotDef::VECTOR_DENSE);
        header.add_seq_type(DataHeader2::NON_SEQ);


        std::vector<DataSample2> samples(batchSize);
        for (int i = 0; i < batchSize; ++i) {
          SlotSample *slotSample = samples[i].add_slots_data();
          slotSample->set_slot_id(0);
          VectorSlot *vec = slotSample->add_vector_slots();
          vec->add_dims(slotDim);
          auto valueBuffer = vec->mutable_values();
          valueBuffer->Reserve(slotDim);
          for (int j = 0; j < slotDim; ++j) {
            valueBuffer->AddAlreadyReserved(distValue(rng));
          }
        }

        mkDir(kTestDir);
        writeData(header, samples, dataCompression);
        DataProvider *provider = createProvider(useGpu, dataCompression, async);
        provider->setSkipShuffle();
        provider->reset();
        int64_t size = 100;
        DataBatch batchOut;
        int actBatchSize = 0;
        do {
          actBatchSize = provider->getNextBatch(size, &batchOut);
        } while (actBatchSize != 0);
        provider->reset();
        LOG(INFO) << "reset.";
        do {
          actBatchSize = provider->getNextBatch(size, &batchOut);
        } while (actBatchSize != 0);
        rmDir(kTestDir);
      }
    }
  }
}

int main(int argc, char** argv) {
  initMain(argc, argv);
  hl_start();
  hl_init(FLAGS_gpu_id);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
