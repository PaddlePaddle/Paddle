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

#pragma once

#include <paddle/utils/PythonUtil.h>
#include "DataFormat.pb.h"
#include "DataProvider.h"

#include <vector>

namespace paddle {

class PyDataProvider : public DataProvider {
 public:
  PyDataProvider(const DataConfig& config,
                 bool useGpu,
                 bool loadDataAll = true);

  virtual void reset();

  // Note this size includes the sequences which are skipped because they
  // are longer than the batch size
  virtual int64_t getSize() {
    LOG(FATAL) << "Not implement yet";
    return -1;
  }
  virtual void shuffle();

  virtual int64_t getNextBatchInternal(int64_t size, DataBatch* batch);

 protected:
  struct ProtoSlot;
  // return false if each each sample is one sequence, i.e., independent
  // of other samples.
  inline bool iidData() const { return isIID_; }

  void parseHeaderData(const std::string& headerData);
  void fillDenseSlot(ProtoSlot& slot, char*& data, const char* dataEnd);
  void fillSparseNonValueSlot(ProtoSlot& slot,
                              char*& data,
                              const char* dataEnd);
  void fillSparseValueSlot(ProtoSlot& slot, char*& data, const char* dataEnd);
  void fillIndexSlot(ProtoSlot& slot, char*& data, const char* dataEnd);
  void fillStringSlot(ProtoSlot& slot, char*& data, const char* dataEnd);
  void fillSlotsByStr(const std::string& samples);
  void handleDenseSlot(ProtoSlot& slot,
                       size_t slotIndex,
                       std::vector<Argument>& cpuArguments);
  void handleSparseNonValueSlot(ProtoSlot& slot,
                                size_t slotIndex,
                                std::vector<Argument>& cpuArguments);
  void handleSparseValueSlot(ProtoSlot& slot,
                             size_t slotIndex,
                             std::vector<Argument>& cpuArguments);
  void handleIndexSlot(ProtoSlot& slot,
                       size_t slotIndex,
                       std::vector<Argument>& cpuArguments);
  void handleStringSlot(ProtoSlot& slot,
                        size_t slotIndex,
                        std::vector<Argument>& cpuArguments);
  void resetSlots();
  void loadData(const std::vector<std::string>& fileList);

 protected:
  struct ProtoSlot {
    SlotDef::SlotType type;
    int dim;
    unsigned int sampleNum;
    unsigned int sequenceNum;
    unsigned int subSequenceNum;
    // Store the data of index type slot
    std::vector<int> indexData;
    // Store the data of dense type slot
    std::vector<real> denseData;
    // Store the data of sparseNonValue type slot
    std::vector<sparse_non_value_t> sparseNonValueData;
    // Store the data of sparseValue type slot
    std::vector<sparse_float_value_t> sparseFloatValueData;
    // Used to store the index of each sample in slot values
    std::vector<int64_t> indices;
    // The starting position of each sequence in samples
    // The last element should be the number of samples
    // If empty, each sample is one sequence.
    std::vector<size_t> sequenceStartPositions;
    // The index id of sequences in slot
    std::vector<int64_t> sampleSequenceIdVec;
    // The starting position of each subsequence in samples
    // The last element should be the number of subsequence
    // If empty, each sequence of sample has no subsequence.
    std::vector<size_t> subSequenceStartPositions;
    // Store the data of string type slot
    std::vector<std::string> strData;
  };
  std::vector<ProtoSlot> slots_;

  PyObjectPtr classInstance_;
  unsigned int batchSize_;
  unsigned int slotNum_;
  // if use sequence, isIID_ equals false, otherwise it is true.
  bool isIID_;
  // The name of python module name
  std::string pyModuleName_;
  // The name of python class name
  std::string pyClassName_;
  // User args set in config
  std::map<std::string, std::string> pyUserArgs_;

  ThreadLocalD<DataBatch> cpuBatch_;
  ThreadLocalD<DataBatch> gpuBatch_;
};

}  // namespace paddle
