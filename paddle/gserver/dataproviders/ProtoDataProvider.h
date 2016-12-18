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

#pragma once

#include <vector>

#include "DataFormat.pb.h"
#include "paddle/utils/Stat.h"

#include "DataProvider.h"
#include "ProtoReader.h"

namespace paddle {

/**
 * @brief Provider data from protobuf data file with each sample
 * specified by proto message
 *
 * DataSample defined in DataFormat.proto.
 *
 * The file format is
 *
 *    header
 *
 *    sample1
 *
 *    sample2
 *
 *    ...
 *
 *    sampleN
 *
 * @note: In the data file, each message is prefixed with its length.
 * The read/write of the protbuf are implemented in ProtoReader.h
 */
class ProtoDataProvider : public DataProvider {
public:
  ProtoDataProvider(const DataConfig& config,
                    bool useGpu,
                    bool loadDataAll = true);
  virtual void reset();

  /**
   * @note this size includes the sequences which are skipped because they
   * are longer than the batch size.
   */
  virtual int64_t getSize() {
    int64_t size = sampleNums_;
    if (usageRatio_ < 1.0f) {
      size = static_cast<int64_t>(size * usageRatio_);
    }
    return size;
  }
  virtual void shuffle();

  void loadData(const std::vector<std::string>& fileList);

  virtual int64_t getNextBatchInternal(int64_t size, DataBatch* batch);

protected:
  /**
   * @brief load protobuf data from a list of file
   * @param[in]  fileName  file name of a file which contains
   * a list of file names
   */
  void loadData(const std::string& fileName);

  /**
   * @brief load protobuf data from file
   * @param[in]  fileName   data file name
   */
  void loadDataFile(const std::string& fileName);
  /** @brief check data header of each data sample
   *  @param[in] header     data header read from protobuf data
   */
  void checkDataHeader(const DataHeader& header);
  /**
   * @brief fill protobuf data into slot_,
   * slot_ is a vector of ProtoSlot in memory.
   * @param[in]  sample     data sample read from protobuf data
   */
  void fillSlots(const DataSample& sample);

  /**
   * @brief return true if each sample is one sequence, i.e., independent
   * of other samples.
   */
  inline bool iidData() const { return sequenceStartPositions_.empty(); }

  /**
   * @brief check that sample is consistent with header_
   */
  void checkSample(const DataSample& sample);

  template <class Op>
  int64_t sequenceLoop(Op op, int64_t size);

  template <class Op>
  int64_t sampleLoop(Op op, int64_t size);

  template <class Op>
  int64_t subSampleLoop(Op op, int64_t size, int slot);

  void showDataStats();

protected:
  struct ProtoVarSlot {
    std::vector<real> data;
    std::vector<int> dims;
  };

  struct ProtoSlot {
    SlotDef::SlotType type;
    int dim;
    std::vector<int> indexData;
    std::vector<real> denseData;
    std::vector<sparse_non_value_t> sparseNonValueData;
    std::vector<sparse_float_value_t> sparseFloatValueData;
    std::vector<int64_t> indices;
    std::vector<int64_t> subIndices;

    std::vector<ProtoVarSlot> varDenseData;
    std::vector<std::vector<int>> varIndices;
    std::vector<std::string> strData;
  };
  DataHeader header_;
  int numVecSlots_;

  std::vector<ProtoSlot> slots_;
  size_t sampleNums_;

  /**
   * The starting position of each sequence in samples.
   * The last element should be num of samples.
   * If empty, each sample is one sequence.
   */
  std::vector<size_t> sequenceStartPositions_;

  int64_t currentSequenceIndex_;

  // The size should be the number of sequences.
  std::vector<size_t> shuffledSequenceIds_;

  ThreadLocalD<DataBatch> cpuBatch_;
  ThreadLocalD<DataBatch> gpuBatch_;

  RWLock lock_;
  std::vector<StatPtr> nnzStats_;  // stats for number of none-zeros entries
};

/**
 * @brief Special use for Proto data: instances should contain sparse-non-value
 * slots
 * and label.
 *
 * @note ProtoSequenceDataProvider treats each SPARSE SLOT as a SEQUENCE
 */
class ProtoSequenceDataProvider : public ProtoDataProvider {
public:
  ProtoSequenceDataProvider(const DataConfig& config,
                            bool useGpu,
                            bool loadDataAll = true);
  ~ProtoSequenceDataProvider() {}
  virtual int64_t getNextBatchInternal(int64_t size, DataBatch* batch);
};

}  // namespace paddle
