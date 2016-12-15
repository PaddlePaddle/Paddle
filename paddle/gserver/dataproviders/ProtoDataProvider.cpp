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

#include "ProtoDataProvider.h"
#include <algorithm>
#include <fstream>
#include <istream>
#include "paddle/utils/StringUtil.h"
#include "paddle/utils/Util.h"

#include "DataProviderGroup.h"
#include "paddle/utils/Logging.h"

DEFINE_double(memory_threshold_on_load_data,
              1.0,
              "stop loading data when memory is not sufficient");

namespace paddle {

REGISTER_DATA_PROVIDER(proto_group, DataProviderGroup<ProtoDataProvider>);
REGISTER_DATA_PROVIDER(proto_sequence_group,
                       DataProviderGroup<ProtoSequenceDataProvider>);

ProtoDataProvider::ProtoDataProvider(const DataConfig& config,
                                     bool useGpu,
                                     bool loadDataAll)
    : DataProvider(config, useGpu), sampleNums_(0), currentSequenceIndex_(0) {
  if (loadDataAll) {
    loadData(config_.files());
  }
}

void ProtoDataProvider::loadData(const std::vector<std::string>& fileList) {
  for (auto& file : fileList) {
    if (FLAGS_memory_threshold_on_load_data < 1.0) {
      double memUsage = getMemoryUsage();
      if (memUsage > FLAGS_memory_threshold_on_load_data) {
        LOG(INFO) << "memUsage is " << memUsage << ", > "
                  << FLAGS_memory_threshold_on_load_data
                  << " therefore SKIP ALL REMAINING file.";
        break;
      }
    }
    LOG(INFO) << "load data file " << file;
    loadDataFile(file);
  }

  if (sequenceStartPositions_.size() == sampleNums_) {
    // This means that each sample is one sequence
    shuffledSequenceIds_.swap(sequenceStartPositions_);
  } else {
    sequenceStartPositions_.push_back(sampleNums_);
    shuffledSequenceIds_.reserve(sequenceStartPositions_.size() - 1);
    for (size_t i = 0; i < sequenceStartPositions_.size() - 1; ++i) {
      shuffledSequenceIds_.push_back(i);
    }
  }

  LOG(INFO) << "read done, num of instance=" << sampleNums_;
  showDataStats();
}

void ProtoDataProvider::loadData(const std::string& fileName) {
  std::vector<std::string> fileList;
  loadFileList(fileName, fileList);
  loadData(fileList);
}

void ProtoDataProvider::checkDataHeader(const DataHeader& header) {
  if (header_.slot_defs_size()) {
    // header_ is already set. Need to check consistency.
    CHECK_EQ(header_.slot_defs_size(), header.slot_defs_size())
        << "Different header";
    for (int i = 0; i < header.slot_defs_size(); ++i) {
      CHECK_EQ(header_.slot_defs(i).type(), header.slot_defs(i).type());
      CHECK_EQ(header_.slot_defs(i).dim(), header.slot_defs(i).dim());
    }
    return;
  }

  // header_ is not set before
  CHECK(header.slot_defs_size()) << "Invalid header: no slot is defined";
  int i;
  for (i = 0; i < header.slot_defs_size(); ++i) {
    if (header.slot_defs(i).type() == SlotDef::INDEX ||
        header.slot_defs(i).type() == SlotDef::VAR_MDIM_INDEX) {
      break;
    }
    constexpr int kBufLen = 100;
    char buf[kBufLen];
    snprintf(buf, kBufLen, "slot%d_nnz", i);
    nnzStats_.push_back(getStat(buf));
  }
  numVecSlots_ = i;

  // Check that INDEX slots are after VECTOR slots
  for (int i = numVecSlots_; i < header.slot_defs_size(); ++i) {
    CHECK(header.slot_defs(i).type() == SlotDef::INDEX ||
          header.slot_defs(i).type() == SlotDef::VAR_MDIM_INDEX);
  }

  slots_.clear();
  slots_.reserve(header.slot_defs_size());
  for (int i = 0; i < header.slot_defs_size(); ++i) {
    slots_.emplace_back();
    slots_.back().type = header.slot_defs(i).type();
    slots_.back().dim = header.slot_defs(i).dim();
    if (SlotDef::VECTOR_SPARSE_NON_VALUE == header.slot_defs(i).type() ||
        SlotDef::VECTOR_SPARSE_VALUE == header.slot_defs(i).type()) {
      slots_.back().indices.push_back(0);
    }
  }

  header_ = header;
}

void ProtoDataProvider::checkSample(const DataSample& sample) {
  CHECK_EQ(numVecSlots_, sample.vector_slots_size());
  CHECK(header_.slot_defs_size() == numVecSlots_ + sample.id_slots_size() ||
        header_.slot_defs_size() == numVecSlots_ + sample.var_id_slots_size());
  for (int i = 0; i < numVecSlots_; ++i) {
    uint32_t dim = header_.slot_defs(i).dim();
    switch (header_.slot_defs(i).type()) {
      case SlotDef::VECTOR_DENSE: {
        CHECK_EQ(static_cast<int>(dim), sample.vector_slots(i).values_size());
        CHECK_EQ(0, sample.vector_slots(i).ids_size());
        break;
      }
      case SlotDef::VECTOR_SPARSE_NON_VALUE: {
        if (0 == sample.vector_slots(i).ids_size()) {
          break;
        }
        CHECK_LT(0, sample.vector_slots(i).ids_size());
        CHECK_EQ(0, sample.vector_slots(i).values_size());
        auto maxId = *std::max_element(sample.vector_slots(i).ids().begin(),
                                       sample.vector_slots(i).ids().end());
        CHECK_GT(dim, maxId);
        break;
      }
      case SlotDef::VECTOR_SPARSE_VALUE: {
        if (0 == sample.vector_slots(i).ids_size()) {
          CHECK_EQ(0, sample.vector_slots(i).values_size());
          break;
        }
        CHECK_LT(0, sample.vector_slots(i).values_size());
        CHECK_GE(static_cast<int>(dim), sample.vector_slots(i).values_size());
        CHECK_EQ(sample.vector_slots(i).values_size(),
                 sample.vector_slots(i).ids_size());
        auto maxId = *std::max_element(sample.vector_slots(i).ids().begin(),
                                       sample.vector_slots(i).ids().end());
        CHECK_GT(dim, maxId);
        break;
      }
      case SlotDef::VAR_MDIM_DENSE: {
        if (static_cast<int>(dim) != 0) {
          CHECK_EQ(static_cast<int>(dim), sample.vector_slots(i).values_size());
          if (sample.vector_slots(i).dims_size() != 0) {
            int totalDim = sample.vector_slots(i).dims(0);
            for (int j = 1; j < sample.vector_slots(i).dims_size(); ++j) {
              totalDim *= sample.vector_slots(i).dims(j);
            }
            CHECK_EQ(static_cast<int>(dim), totalDim);
          }
        } else {
          CHECK_NE(sample.vector_slots(i).dims_size(), 0);
          int totalDim = sample.vector_slots(i).dims(0);
          for (int j = 1; j < sample.vector_slots(i).dims_size(); ++j) {
            totalDim *= sample.vector_slots(i).dims(j);
          }
          CHECK_EQ(totalDim, sample.vector_slots(i).values_size());
        }
        break;
      }
      case SlotDef::STRING: {
        CHECK_EQ(static_cast<int>(1), sample.vector_slots(i).strs_size());
        CHECK_EQ(0, sample.vector_slots(i).ids_size());
        CHECK_EQ(0, sample.vector_slots(i).values_size());
        break;
      }
      default:
        LOG(FATAL) << "BUG: Should not reach here";
    }
  }
  for (int i = numVecSlots_; i < header_.slot_defs_size(); ++i) {
    if (header_.slot_defs(i).type() != SlotDef::VAR_MDIM_INDEX) {
      uint32_t id = sample.id_slots(i - numVecSlots_);
      if (id == -1U) continue;
      CHECK_LT(id, header_.slot_defs(i).dim());
    } else {
      for (int j = 0; j < sample.var_id_slots(i - numVecSlots_).ids_size();
           ++j) {
        uint32_t id = sample.var_id_slots(i - numVecSlots_).ids(j);
        CHECK_LT(id, header_.slot_defs(i).dim());
      }
    }
  }
}

void ProtoDataProvider::loadDataFile(const std::string& fileName) {
  std::ifstream is(fileName);
  CHECK(is) << "Fail to open " << fileName;
  bool dataCompression = str::endsWith(fileName, ".gz");
  std::unique_ptr<ProtoReader> reader(new ProtoReader(&is, dataCompression));
  CHECK(reader) << "Fail to create proto data input stream";

  DataHeader header;
  CHECK(reader->read(&header));
  checkDataHeader(header);

  DataSample sample;
  do {
    if (!reader->read(&sample)) {
      break;
    }
    checkSample(sample);
    if (sample.is_beginning()) {
      sequenceStartPositions_.push_back(sampleNums_);
    }
    fillSlots(sample);
    ++sampleNums_;
  } while (true);

  CHECK(is.eof()) << "Fail to read file";
  reader.reset(nullptr);
  is.close();
}

// checkSample has done before, no check here
void ProtoDataProvider::fillSlots(const DataSample& sample) {
  for (size_t i = 0; i < slots_.size(); ++i) {
    auto& slot = slots_[i];
    int dim = slot.dim;
    switch (slot.type) {
      case SlotDef::VECTOR_DENSE: {
        size_t oldSize = slot.denseData.size();
        slot.denseData.resize(oldSize + dim);
        const float* values = sample.vector_slots(i).values().data();
#ifdef PADDLE_TYPE_DOUBLE
        std::copy(values, values + dim, slot.denseData.begin() + oldSize);
#else
        memcpy(slot.denseData.data() + oldSize, values, sizeof(real) * dim);
#endif
        break;
      }
      case SlotDef::VECTOR_SPARSE_NON_VALUE: {
        int slotSize = sample.vector_slots(i).ids_size();
        int subSlotSize = 0;
        int id = 0;  // the slot id
        // find whether this vector_slots has subseq. If not has subseq,
        // subSlotSize = 0.
        for (id = 0; id < sample.subseq_slots_size(); id++) {
          if (sample.subseq_slots(id).slot_id() == i) {
            subSlotSize = sample.subseq_slots(id).lens_size();
            break;
          }
        }
        if (subSlotSize && slot.subIndices.size() == 0UL) {
          // If has subSeq, the first element of subIndices = 0.
          slot.subIndices.push_back(0);
        }
        if (slotSize == 0UL) {
          // if has no id, new indices = old indices.
          slot.indices.push_back(slot.indices.back());
          // if has subSeq, new subIndices = old subIndices.
          if (slot.subIndices.size()) {
            slot.subIndices.push_back(slot.subIndices.back());
          }
          break;
        }
        slot.sparseNonValueData.resize(slot.indices.back() + slotSize);
        const unsigned int* ids = sample.vector_slots(i).ids().data();
        memcpy(slot.sparseNonValueData.data() + slot.indices.back(),
               ids,
               sizeof(*ids) * slotSize);
        slot.indices.push_back(slot.indices.back() + slotSize);
        if (subSlotSize) {
          for (int ii = 0; ii < subSlotSize; ++ii) {
            slot.subIndices.push_back(slot.subIndices.back() +
                                      sample.subseq_slots(id).lens(ii));
          }
        }
        break;
      }
      case SlotDef::VECTOR_SPARSE_VALUE: {
        if (0 == sample.vector_slots(i).ids_size()) {
          slot.indices.push_back(slot.indices.back());
          break;
        }
        int slotSize = sample.vector_slots(i).ids_size();
        slot.sparseFloatValueData.resize(slot.indices.back() + slotSize);
        const unsigned int* ids = sample.vector_slots(i).ids().data();
        const float* values = sample.vector_slots(i).values().data();
        for (int ii = 0; ii < slotSize; ++ii) {
          slot.sparseFloatValueData[slot.indices.back() + ii].col = ids[ii];
          slot.sparseFloatValueData[slot.indices.back() + ii].value =
              values[ii];
        }
        slot.indices.push_back(slot.indices.back() + slotSize);
        break;
      }
      case SlotDef::INDEX: {
        slot.indexData.push_back(sample.id_slots(i - numVecSlots_));
        break;
      }
      case SlotDef::VAR_MDIM_DENSE: {
        size_t oldSize = slot.varDenseData.size();
        slot.varDenseData.resize(oldSize + 1);
        size_t varDim = sample.vector_slots(i).values_size();
        slot.varDenseData[oldSize].data.resize(varDim);
        const float* values = sample.vector_slots(i).values().data();
#ifdef PADDLE_TYPE_DOUBLE
        std::copy(
            values, values + varDim, slot.varDenseData[oldSize].data.data());
#else
        memcpy(slot.varDenseData[oldSize].data.data(),
               values,
               sizeof(real) * varDim);
#endif
        slot.varDenseData[oldSize].dims.resize(
            sample.vector_slots(i).dims_size());
        memcpy(slot.varDenseData[oldSize].dims.data(),
               sample.vector_slots(i).dims().data(),
               sizeof(uint32_t) * sample.vector_slots(i).dims_size());
        break;
      }
      case SlotDef::VAR_MDIM_INDEX: {
        size_t oldSize = slot.varIndices.size();
        slot.varIndices.resize(oldSize + 1);
        size_t varDim = sample.var_id_slots(i - numVecSlots_).ids_size();
        slot.varIndices[oldSize].resize(varDim);
        memcpy(slot.varIndices[oldSize].data(),
               sample.var_id_slots(i - numVecSlots_).ids().data(),
               sizeof(uint32_t) * varDim);
        break;
      }
      case SlotDef::STRING: {
        slot.strData.push_back(sample.vector_slots(i).strs(0));
        break;
      }
    }
  }
}

void ProtoDataProvider::showDataStats() {
  std::ostringstream oss;
  for (size_t i = 0; i < slots_.size(); ++i) {
    auto& slot = slots_[i];
    if (slot.type == SlotDef::VECTOR_SPARSE_NON_VALUE) {
      size_t nnz = slot.sparseNonValueData.size();
      oss << "slot" << i << ":avgNNZ=" << ((double)nnz / sampleNums_) << "; ";
    } else if (slot.type == SlotDef::VECTOR_SPARSE_VALUE) {
      size_t nnz = slot.sparseFloatValueData.size();
      oss << "slot" << i << ":avgNNZ=" << ((double)nnz / sampleNums_) << "; ";
    }
  }
  LOG(INFO) << oss.str();
}

void ProtoDataProvider::reset() {
  currentSequenceIndex_ = 0;
  if (!skipShuffle_) {
    shuffle();
  }

  DataProvider::reset();
}

void ProtoDataProvider::shuffle() {
  std::shuffle(shuffledSequenceIds_.begin(),
               shuffledSequenceIds_.end(),
               ThreadLocalRandomEngine::get());
}

/*
  Loop through sequences starting from currentSequenceIndex_
  for at most size samples. For each sequence ranging from [begin, end),
  op(begin, end) will be called.

  return the number of sequences scanned
*/
template <class Op>
int64_t ProtoDataProvider::sequenceLoop(Op op, int64_t size) {
  int64_t sz = 0;
  size_t i;
  size_t sequenceCount = shuffledSequenceIds_.size();
  if (usageRatio_ < 1.0f) {
    sequenceCount = static_cast<int64_t>(sequenceCount * usageRatio_);
  }
  for (i = currentSequenceIndex_; i < sequenceCount; ++i) {
    size_t id = shuffledSequenceIds_[i];
    int64_t begin = sequenceStartPositions_[id];
    int64_t end = sequenceStartPositions_[id + 1];
    int64_t len = end - begin;
    if (sz + len > size && sz > 0) break;
    sz += len;
    op(begin, end);
  }
  return i - currentSequenceIndex_;
}

/*
  Loop through sequences starting from currentSequenceIndex_
  for at most size samples. For each sample of each sequence at position
  pos, op(pos) will be called.

  return the number of sequences scanned
*/
template <class Op>
int64_t ProtoDataProvider::sampleLoop(Op op, int64_t size) {
  if (iidData()) {
    size = std::min<int64_t>(sampleNums_ - currentSequenceIndex_, size);
    for (int64_t i = currentSequenceIndex_; i < currentSequenceIndex_ + size;
         ++i) {
      size_t pos = shuffledSequenceIds_[i];
      op(pos);
    }
    return size;
  } else {
    auto f = [op](int64_t begin, int64_t end) {
      for (int64_t pos = begin; pos < end; ++pos) {
        op(pos);
      }
    };
    return sequenceLoop(f, size);
  }
}

/*
  Loop through sub-sequences starting from currentSequenceIndex_
  for at most size samples. For each sample of each sub-sequence at position
  pos, op(pos) will be called.

  return the number of sub-sequences scanned
*/
template <class Op>
int64_t ProtoDataProvider::subSampleLoop(Op op, int64_t size, int slot) {
  CHECK(iidData()) << "subSampleLoop only accepts iid data";
  size = std::min<int64_t>(sampleNums_ - currentSequenceIndex_, size);
  int subSize = 0;
  for (int64_t i = currentSequenceIndex_; i < currentSequenceIndex_ + size;
       ++i) {
    size_t pos = shuffledSequenceIds_[i];
    int64_t* indexs = slots_[slot].indices.data();
    int64_t* subIndexs = slots_[slot].subIndices.data();
    int64_t subSeqStart = 0;
    int64_t subSeqEnd = 0;
    for (int j = 0; j < (int)slots_[slot].subIndices.size(); j++) {
      if (subIndexs[j] == indexs[pos]) {
        subSeqStart = j;
        if (subIndexs[pos] == subIndexs[pos + 1]) {
          subSeqEnd = j + 1;
          break;
        }
      } else if (subIndexs[j] == indexs[pos + 1]) {
        subSeqEnd = j;
        break;
      }
    }
    for (int j = subSeqStart; j < subSeqEnd; j++) {
      op(j);
    }
    subSize += subSeqEnd - subSeqStart;
  }
  return subSize;
}

int64_t ProtoDataProvider::getNextBatchInternal(int64_t size,
                                                DataBatch* batch) {
  int64_t numSequences = 0;  // actual number of sequences in the batch

  // the number of sequences scanned, including those skipped because too long
  int64_t numScannedSeqs = 0;
  std::lock_guard<RWLock> guard(lock_);
  if (iidData()) {
    size = std::min<int64_t>(getSize() - currentSequenceIndex_, size);
    numScannedSeqs = numSequences = size;
  } else {
    int64_t sz = 0;
    auto op = [&sz, &numSequences](int64_t begin, int64_t end) {
      ++numSequences;
      sz += end - begin;
    };
    numScannedSeqs = sequenceLoop(op, size);
    VLOG_IF(1, numScannedSeqs > numSequences)
        << numScannedSeqs - numSequences
        << " sequences are skipped because longer than " << size;
    size = sz;
  }
  if (size <= 0) return 0;

  DataBatch& cpuBatch = *cpuBatch_;
  std::vector<Argument>& cpuArguments = cpuBatch.getStreams();
  cpuBatch.setSize(size);
  cpuArguments.resize(header_.slot_defs_size());

  if (!iidData()) {
    ICpuGpuVector::resizeOrCreate(cpuArguments[0].sequenceStartPositions,
                                  numSequences + 1,
                                  /* useGpu= */ false);
    int* buf = cpuArguments[0].sequenceStartPositions->getMutableData(false);
    int pos = 0;
    int i = 0;
    auto op = [buf, &pos, &i](int64_t begin, int64_t end) {
      buf[i] = pos;
      pos += end - begin;
      ++i;
    };
    sequenceLoop(op, size);
    buf[i] = size;
    for (size_t slot = 1; slot < cpuArguments.size(); ++slot) {
      cpuArguments[slot].sequenceStartPositions =
          cpuArguments[0].sequenceStartPositions;
    }
  }

  for (int slot = 0; slot < header_.slot_defs_size(); ++slot) {
    size_t dim = header_.slot_defs(slot).dim();
    SlotDef::SlotType slotType = header_.slot_defs(slot).type();

    std::vector<int64_t> dataPos;
    dataPos.reserve(size);
    auto op = [this, &dataPos](int64_t pos) { dataPos.push_back(pos); };
    sampleLoop(op, size);

    switch (slotType) {
      case SlotDef::VECTOR_DENSE: {
        Matrix::resizeOrCreate(cpuArguments[slot].value,
                               size,
                               dim,
                               false,   // trans = false
                               false);  // useGpu = false
        real* buf = cpuArguments[slot].value->getData();
        for (int i = 0; i < size; ++i) {
          memcpy(buf + i * dim,
                 slots_[slot].denseData.data() + dataPos[i] * dim,
                 sizeof(real) * dim);
        }
        break;
      }
      case SlotDef::VECTOR_SPARSE_NON_VALUE: {
        if (!(cpuArguments[slot].value)) {
          cpuArguments[slot].value =
              Matrix::createSparseMatrix(size,
                                         dim,
                                         size /*DEFAULT_AVG_WIDTH = 1*/,
                                         NO_VALUE,
                                         SPARSE_CSR,
                                         false,
                                         useGpu_);
        }
        auto mat = cpuArguments[slot].value;
        mat->resize(size, dim);
        if (std::dynamic_pointer_cast<GpuSparseMatrix>(mat)) {
          std::dynamic_pointer_cast<GpuSparseMatrix>(mat)->copyFrom(
              dataPos.data(),
              slots_[slot].indices.data(),
              slots_[slot].sparseNonValueData.data(),
              HPPL_STREAM_1);
        } else if (std::dynamic_pointer_cast<CpuSparseMatrix>(mat)) {
          std::dynamic_pointer_cast<CpuSparseMatrix>(mat)->copyFrom(
              dataPos.data(),
              slots_[slot].indices.data(),
              slots_[slot].sparseNonValueData.data());
        } else {
          LOG(FATAL) << "Not Supported";
        }
        size_t numElements = 0;
        for (auto pos : dataPos) {
          numElements +=
              slots_[slot].indices[pos + 1] - slots_[slot].indices[pos];
        }
        nnzStats_[slot]->addSample(numElements);

        break;
      }
      case SlotDef::VECTOR_SPARSE_VALUE: {
        if (!(cpuArguments[slot].value)) {
          cpuArguments[slot].value =
              Matrix::createSparseMatrix(size,
                                         dim,
                                         size /*DEFAULT_AVG_WIDTH = 1*/,
                                         FLOAT_VALUE,
                                         SPARSE_CSR,
                                         false,
                                         useGpu_);
        }
        auto mat = cpuArguments[slot].value;
        mat->resize(size, dim);
        if (std::dynamic_pointer_cast<GpuSparseMatrix>(mat)) {
          std::dynamic_pointer_cast<GpuSparseMatrix>(mat)->copyFrom(
              dataPos.data(),
              slots_[slot].indices.data(),
              slots_[slot].sparseFloatValueData.data(),
              HPPL_STREAM_1);
        } else if (std::dynamic_pointer_cast<CpuSparseMatrix>(mat)) {
          std::dynamic_pointer_cast<CpuSparseMatrix>(mat)->copyFrom(
              dataPos.data(),
              slots_[slot].indices.data(),
              slots_[slot].sparseFloatValueData.data());
        } else {
          LOG(FATAL) << "Not Supported";
        }
        break;
      }
      case SlotDef::INDEX: {
        IVector::resizeOrCreate(cpuArguments[slot].ids,
                                size,
                                /*  useGpu= */ false);
        int* buf = cpuArguments[slot].ids->getData();
        for (int i = 0; i < size; ++i) {
          buf[i] = slots_[slot].indexData[dataPos[i]];
        }
        break;
      }
      case SlotDef::VAR_MDIM_DENSE: {
        CHECK_EQ(size, 1);
        auto mat = cpuArguments[slot].value;
        size_t totalDim = slots_[slot].varDenseData[dataPos[0]].data.size();

        CHECK_EQ(slots_[slot].varDenseData[dataPos[0]].dims.size(), size_t(3));
        size_t height, width, depth, oldWidth;
        /* dims[2] is depth, will be changed to dims[0] in future */
        depth = slots_[slot].varDenseData[dataPos[0]].dims[2];
        height = slots_[slot].varDenseData[dataPos[0]].dims[1];
        width = slots_[slot].varDenseData[dataPos[0]].dims[0];
        oldWidth = width;
        /* process the undesirable sample */
        if (oldWidth < height) {
          width = height;
        }
        cpuArguments[slot].setFrameHeight(height);
        cpuArguments[slot].setFrameWidth(width);

        if (oldWidth < height) {
          totalDim = width * height * depth;
        }
        Matrix::resizeOrCreate(cpuArguments[slot].value,
                               size,
                               totalDim,
                               false,   // trans = false
                               false);  // useGpu = false
        real* buf = cpuArguments[slot].value->getData();
        cpuArguments[slot].value->zeroMem();
        if (oldWidth < height) {
          real* srcBuf = slots_[slot].varDenseData[dataPos[0]].data.data();
          for (size_t i = 0; i < depth; i++) {
            for (size_t j = 0; j < height; j++) {
              for (size_t k = 0; k < oldWidth; k++) {
                buf[i * height * width + j * width + k] =
                    srcBuf[i * height * oldWidth + j * oldWidth + k];
              }
            }
          }
        } else {
          memcpy(buf,
                 slots_[slot].varDenseData[dataPos[0]].data.data(),
                 sizeof(real) * totalDim);
        }
        ICpuGpuVector::resizeOrCreate(cpuArguments[slot].sequenceStartPositions,
                                      size + 1, /* size == 1 currently */
                                      /* useGpu= */ false);
        int* bufStarts =
            cpuArguments[slot].sequenceStartPositions->getMutableData(false);
        bufStarts[0] = 0;
        bufStarts[1] = 1;
        break;
      }
      case SlotDef::VAR_MDIM_INDEX: {
        CHECK_EQ(size, 1);
        size_t totalDim = slots_[slot].varIndices[dataPos[0]].size();
        IVector::resizeOrCreate(cpuArguments[slot].ids,
                                totalDim,
                                /*  useGpu= */ false);
        int* buf = cpuArguments[slot].ids->getData();
        memcpy(buf,
               slots_[slot].varIndices[dataPos[0]].data(),
               sizeof(int) * totalDim);

        ICpuGpuVector::resizeOrCreate(cpuArguments[slot].sequenceStartPositions,
                                      size + 1, /* size == 1 currently */
                                      /* useGpu= */ false);
        int* bufStarts =
            cpuArguments[slot].sequenceStartPositions->getMutableData(false);
        bufStarts[0] = 0;
        /* we expand the convolutinal feature map to a sequence data,
         * so there should be a corresponding sequence labels */
        bufStarts[1] = totalDim;
        break;
      }
      case SlotDef::STRING: {
        if (cpuArguments[slot].strs) {
          cpuArguments[slot].strs->resize(size);
        } else {
          cpuArguments[slot].strs =
              std::make_shared<std::vector<std::string>>(size);
        }
        for (int i = 0; i < size; ++i) {
          (*cpuArguments[slot].strs)[i] = slots_[slot].strData[dataPos[i]];
        }
        break;
      }
    }
  }

  if (useGpu_) {
    std::vector<Argument>& cpuArguments = cpuBatch.getStreams();
    DataBatch& gpuBatch = *gpuBatch_;
    std::vector<Argument>& gpuArguments = gpuBatch.getStreams();
    gpuArguments.resize(cpuArguments.size());
    gpuBatch.setSize(size);
    for (int i = 0; i < header_.slot_defs_size(); ++i) {
      SlotDef::SlotType slotType = header_.slot_defs(i).type();
      if (SlotDef::VECTOR_SPARSE_VALUE == slotType ||
          SlotDef::VECTOR_SPARSE_NON_VALUE == slotType) {
        gpuArguments[i] = cpuArguments[i];
        gpuArguments[i].sequenceStartPositions =
            cpuArguments[i].sequenceStartPositions;
      } else {
        gpuArguments[i].resizeAndCopyFrom(
            cpuArguments[i], useGpu_, HPPL_STREAM_1);
      }
    }
    hl_stream_synchronize(HPPL_STREAM_1);
    *batch = gpuBatch;
  } else {
    *batch = cpuBatch;
  }

  currentSequenceIndex_ += numScannedSeqs;

  return batch->getSize();
}

ProtoSequenceDataProvider::ProtoSequenceDataProvider(const DataConfig& config,
                                                     bool useGpu,
                                                     bool loadDataAll)
    : ProtoDataProvider(config, useGpu, loadDataAll) {}

int64_t ProtoSequenceDataProvider::getNextBatchInternal(int64_t size,
                                                        DataBatch* batch) {
  CHECK(iidData()) << "ProtoSequenceDataProvider only accepts iid data";
  int64_t numSequences = 0;  // actual number of sequences in the batch

  // the number of sequences scanned, including those skipped because too long
  int64_t numScannedSeqs = 0;
  std::lock_guard<RWLock> guard(lock_);
  size = std::min<int64_t>(getSize() - currentSequenceIndex_, size);
  numScannedSeqs = numSequences = size;
  if (size <= 0) return 0;

  DataBatch& cpuBatch = *cpuBatch_;
  std::vector<Argument>& cpuArguments = cpuBatch.getStreams();
  cpuBatch.setSize(size);
  cpuArguments.resize(header_.slot_defs_size());

  for (int slot = 0; slot < header_.slot_defs_size(); ++slot) {
    SlotDef::SlotType slotType = header_.slot_defs(slot).type();

    std::vector<int64_t> dataPos;
    dataPos.reserve(size);
    auto op = [this, &dataPos](int64_t pos) { dataPos.push_back(pos); };
    sampleLoop(op, size);

    // current slot: sequenceStartPositions
    ICpuGpuVector::resizeOrCreate(cpuArguments[slot].sequenceStartPositions,
                                  size + 1,
                                  /* useGpu= */ false);

    switch (slotType) {
      case SlotDef::VECTOR_SPARSE_VALUE:
      case SlotDef::VAR_MDIM_DENSE:
      case SlotDef::VAR_MDIM_INDEX: {
        LOG(FATAL) << "ProtoSequenceDataProvider only support"
                   << " VECTOR_DENSE, VECTOR_SPARSE_NON_VALUE and INDEX slots";
        break;
      }
      case SlotDef::VECTOR_SPARSE_NON_VALUE: {
        // copy to IDS, not value
        // pointers used in current slot
        sparse_non_value_t* data = slots_[slot].sparseNonValueData.data();
        int64_t* indexs = slots_[slot].indices.data();
        int64_t* seqs = dataPos.data();

        // current slot: i need size instances. what is the total length?
        int totalFeatureInCurrentSlot = 0;
        for (int ins = 0; ins < size; ins++) {
          int64_t currInsId = seqs[ins];
          totalFeatureInCurrentSlot +=
              indexs[currInsId + 1] - indexs[currInsId];
          // special: if current instance has NO feature in current slot
          if (indexs[currInsId + 1] == indexs[currInsId]) {
            totalFeatureInCurrentSlot++;
          }
        }
        // done

        // current slot: ids
        IVector::resizeOrCreate(cpuArguments[slot].ids,
                                totalFeatureInCurrentSlot,
                                /* useGpu= */ false);

        // where to write
        int* currPosOfArgumentId = cpuArguments[slot].ids->getData();
        int* currPosOfArgumentSeqStart =
            cpuArguments[slot].sequenceStartPositions->getMutableData(false);
        int allSequenceLength = 0;
        currPosOfArgumentSeqStart[0] = 0;
        // for each instance, copy data and fill sequence positions
        for (int instance = 0; instance < size; instance++) {
          int64_t currInstanceId = seqs[instance];
          int64_t currInstanceLength =
              indexs[currInstanceId + 1] - indexs[currInstanceId];
          sparse_non_value_t* currInstanceData = data + indexs[currInstanceId];
          // write sequenceStartPositions
          allSequenceLength += currInstanceLength;
          currPosOfArgumentSeqStart[instance + 1] = allSequenceLength;
          // copy features
          for (int featCopier = 0; featCopier < currInstanceLength;
               featCopier++) {
            currPosOfArgumentId[featCopier] = currInstanceData[featCopier].col;
          }
          currPosOfArgumentId += currInstanceLength;
          // special: if current instance has NO feature in current slot
          if (currInstanceLength == 0) {
            allSequenceLength++;
            currPosOfArgumentSeqStart[instance + 1] = allSequenceLength;
            currPosOfArgumentId[0] = -1;
            currPosOfArgumentId++;
          }
          // done
        }
        if (slots_[slot].subIndices.size()) {
          std::vector<int64_t> dataSubPos;
          auto op = [this, &dataSubPos](int64_t pos) {
            dataSubPos.push_back(pos);
          };
          int subSize = subSampleLoop(op, size, slot);
          ICpuGpuVector::resizeOrCreate(
              cpuArguments[slot].subSequenceStartPositions, subSize + 1, false);
          int* currPosOfArgumentSubSeqStart =
              cpuArguments[slot].subSequenceStartPositions->getMutableData(
                  false);
          int64_t* subSeqs = dataSubPos.data();
          int64_t* subIndexs = slots_[slot].subIndices.data();
          int allSubSequenceLength = 0;
          currPosOfArgumentSubSeqStart[0] = 0;
          // for each instance, compute sub-sequence number
          for (int instance = 0; instance < subSize; instance++) {
            int64_t currSubInstanceId = subSeqs[instance];
            int64_t currSubInstanceLength =
                subIndexs[currSubInstanceId + 1] - subIndexs[currSubInstanceId];
            // write subSequenceStartPositions
            allSubSequenceLength += currSubInstanceLength;
            currPosOfArgumentSubSeqStart[instance + 1] = allSubSequenceLength;
            // special: if current instance has NO feature in current slot
            if (currSubInstanceLength == 0) {
              allSubSequenceLength++;
              currPosOfArgumentSubSeqStart[instance + 1] = allSubSequenceLength;
            }
          }
          cpuArguments[slot].checkSubset();
        }
        break;
      }
      case SlotDef::INDEX: {
        // label slot
        IVector::resizeOrCreate(cpuArguments[slot].ids,
                                size,
                                /* useGpu= */ false);
        // fill labels
        int* buf = cpuArguments[slot].ids->getData();
        for (int i = 0; i < size; ++i) {
          buf[i] = slots_[slot].indexData[dataPos[i]];
        }
        // label HAS sequence structure
        cpuArguments[slot].sequenceStartPositions->fillSequence(false);
        break;
      }
      case SlotDef::VECTOR_DENSE: {
        // copy values
        size_t dim = header_.slot_defs(slot).dim();
        Matrix::resizeOrCreate(cpuArguments[slot].value,
                               size,
                               dim,
                               false,   // trans = false
                               false);  // useGpu = false
        real* buf = cpuArguments[slot].value->getData();
        for (int i = 0; i < size; ++i) {
          memcpy(buf + i * dim,
                 slots_[slot].denseData.data() + dataPos[i] * dim,
                 sizeof(real) * dim);
        }
        // sequence structure
        cpuArguments[slot].sequenceStartPositions->fillSequence(false);
        break;
      }
      default: { LOG(FATAL) << "should not reach here"; }
    }
  }

  if (useGpu_) {
    std::vector<Argument>& cpuArguments = cpuBatch.getStreams();
    DataBatch& gpuBatch = *gpuBatch_;
    std::vector<Argument>& gpuArguments = gpuBatch.getStreams();
    gpuArguments.resize(cpuArguments.size());
    gpuBatch.setSize(size);
    for (size_t i = 0; i < cpuArguments.size(); ++i) {
      gpuArguments[i].resizeAndCopyFrom(
          cpuArguments[i], useGpu_, HPPL_STREAM_1);
    }
    hl_stream_synchronize(HPPL_STREAM_1);
    *batch = gpuBatch;
  } else {
    *batch = cpuBatch;
  }

  currentSequenceIndex_ += numScannedSeqs;
  return batch->getSize();
}

}  // namespace paddle
