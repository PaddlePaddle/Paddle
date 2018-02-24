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

#include "PyDataProvider.h"
#include "paddle/utils/Common.h"
#include "paddle/utils/PythonUtil.h"
#include "paddle/utils/Util.h"

namespace paddle {

#ifndef PADDLE_NO_PYTHON
REGISTER_DATA_PROVIDER(py, PyDataProvider);
#endif

PyDataProvider::PyDataProvider(const DataConfig& config,
                               bool useGpu,
                               bool loadDataAll)
    : DataProvider(config, useGpu), batchSize_(0) {
  PyGuard guard;
  pyModuleName_ = config_.load_data_module();
  pyClassName_ = config_.load_data_object();
  if (config_.load_data_args() != "") {
    pyUserArgs_["load_data_args"] = config_.load_data_args();
  }

  if (loadDataAll) {
    std::vector<std::string> fileList;
    if (!config_.files().empty()) {
      loadFileList(config_.files(), fileList);
    }
    loadData(fileList);
  }
}

void PyDataProvider::loadData(const std::vector<std::string>& fileList) {
  VLOG(1) << "module:" << pyModuleName_ << " class:" << pyClassName_;
  classInstance_ =
      createPythonClass(pyModuleName_, pyClassName_, fileList, pyUserArgs_);
  CHECK(classInstance_) << "Create class instance failed.";
  PyObjectPtr obj(PyObject_CallMethod(
      classInstance_.get(), const_cast<char*>("getHeader"), NULL));
  CHECK_PY(obj) << "Call function getHeader failed.";
  std::string headerInfo =
      std::string(PyString_AsString(obj.get()), PyString_Size(obj.get()));
  parseHeaderData(headerInfo);
  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
}

void PyDataProvider::parseHeaderData(const std::string& headerData) {
  char* pHeader = const_cast<char*>(headerData.c_str());
  char* pHeaderEnd = pHeader + headerData.size();
  slotNum_ = readT<unsigned int>(pHeader, pHeaderEnd);
  unsigned int useSequenceFlag = readT<unsigned int>(pHeader, pHeaderEnd);
  isIID_ = useSequenceFlag != 1;
  slots_.clear();
  slots_.reserve(slotNum_);
  for (size_t i = 0; i < slotNum_; ++i) {
    unsigned int slotType = readT<unsigned int>(pHeader, pHeaderEnd);
    unsigned int slotDim = readT<unsigned int>(pHeader, pHeaderEnd);
    slots_.emplace_back();
    slots_.back().dim = slotDim;
    slots_.back().type = static_cast<SlotDef_SlotType>(slotType);
  }
}

void PyDataProvider::resetSlots() {
  for (auto& slot : slots_) {
    slot.indexData.clear();
    slot.denseData.clear();
    slot.sparseNonValueData.clear();
    slot.sparseFloatValueData.clear();
    slot.indices.clear();
    slot.sequenceStartPositions.clear();
    slot.sampleSequenceIdVec.clear();
    slot.subSequenceStartPositions.clear();
    slot.strData.clear();
  }
}

void PyDataProvider::fillDenseSlot(ProtoSlot& slot,
                                   char*& data,
                                   const char* dataEnd) {
  unsigned int dim = slot.dim;
  slot.sampleNum = readT<unsigned int>(data, dataEnd);
  slot.denseData.resize(slot.sampleNum * dim);
#ifdef PADDLE_TYPE_DOUBLE
  CHECK_LE(data + sizeof(real) * dim * slot.sampleNum, dataEnd)
      << "std::copy data is out of range";
  // PyDataProvider always provide data in float
  float* dat = reinterpret_cast<float*>(data);
  std::copy(dat, dat + slot.sampleNum * dim, slot.denseData.begin());
#else
  memcpyWithCheck(slot.denseData.data(),
                  data,
                  sizeof(real) * dim * slot.sampleNum,
                  dataEnd);
#endif
  // PyDataProvider always provide data in float
  data += sizeof(float) * dim * slot.sampleNum;
}

void PyDataProvider::fillSparseNonValueSlot(ProtoSlot& slot,
                                            char*& data,
                                            const char* dataEnd) {
  slot.sampleNum = readT<unsigned int>(data, dataEnd);
  unsigned int* indexPtr = (unsigned int*)data;
  CHECK_LE(data + sizeof(unsigned int) * slot.sampleNum, dataEnd)
      << "Vector assign value is out of range";
  slot.indices.assign(indexPtr, indexPtr + slot.sampleNum);
  data += sizeof(unsigned int) * slot.sampleNum;
  unsigned int length = 0;
  length = readT<unsigned int>(data, dataEnd);
  slot.indices.push_back(length);
  slot.sparseNonValueData.resize(length);
  memcpyWithCheck(slot.sparseNonValueData.data(),
                  data,
                  sizeof(unsigned int) * length,
                  dataEnd);
  data += sizeof(unsigned int) * length;
}

void PyDataProvider::fillSparseValueSlot(ProtoSlot& slot,
                                         char*& data,
                                         const char* dataEnd) {
  slot.sampleNum = readT<unsigned int>(data, dataEnd);
  unsigned int* indexPtr = (unsigned int*)data;
  CHECK_LE(data + sizeof(unsigned int) * slot.sampleNum, dataEnd)
      << "Vector assign value is out of range";
  slot.indices.assign(indexPtr, indexPtr + slot.sampleNum);
  data += sizeof(unsigned int) * slot.sampleNum;
  unsigned int length = 0;
  length = readT<unsigned int>(data, dataEnd);
  unsigned int* colPtr = reinterpret_cast<unsigned int*>(data);
  CHECK_LE(data + sizeof(unsigned int) * length, dataEnd)
      << "Data is out of range";
  data += sizeof(unsigned int) * length;
  size_t colLen = readT<unsigned int>(data, dataEnd);
  CHECK_EQ(colLen, length);
  float* valuePtr = reinterpret_cast<float*>(data);
  CHECK_LE(data + sizeof(real) * length, dataEnd) << "Data is out of range";
  data += sizeof(real) * length;
  slot.indices.push_back(length);
  slot.sparseFloatValueData.resize(length);
  for (unsigned int ii = 0; ii < length; ++ii) {
    slot.sparseFloatValueData[ii].col = colPtr[ii];
    slot.sparseFloatValueData[ii].value = valuePtr[ii];
  }
}

void PyDataProvider::fillIndexSlot(ProtoSlot& slot,
                                   char*& data,
                                   const char* dataEnd) {
  slot.sampleNum = readT<unsigned int>(data, dataEnd);
  CHECK_LE(data + sizeof(unsigned int) * slot.sampleNum, dataEnd)
      << "Vector assign is out of range";
  slot.indexData.assign(reinterpret_cast<int*>(data),
                        reinterpret_cast<int*>(data) + slot.sampleNum);
  data += sizeof(unsigned int) * slot.sampleNum;
}

void PyDataProvider::fillStringSlot(ProtoSlot& slot,
                                    char*& data,
                                    const char* dataEnd) {
  slot.sampleNum = readT<unsigned int>(data, dataEnd);
  for (unsigned int i = 0; i < slot.sampleNum; ++i) {
    size_t len = readT<uint32_t>(data, dataEnd);
    auto str_begin = data;
    data += len;
    CHECK_LE(data, dataEnd) << "Data is out of range";
    slot.strData.emplace_back(str_begin, len);
  }
}

void PyDataProvider::fillSlotsByStr(const std::string& samples) {
  char* data = const_cast<char*>(samples.c_str());
  char* dataEnd = data + samples.size();
  batchSize_ = readT<unsigned int>(data, dataEnd);
  if (0 == batchSize_) {
    return;
  }

  for (size_t j = 0; j < slotNum_; ++j) {
    auto& slot = slots_[j];
    CHECK(SlotDef::INDEX >= slot.type || SlotDef::STRING == slot.type)
        << " Slot type:" << slot.type << " is out of range.";
    CHECK_GE(slot.type, SlotDef::VECTOR_DENSE) << " Slot type:" << slot.type
                                               << " is out of range.";
    switch (slot.type) {
      case SlotDef::VECTOR_DENSE:
        fillDenseSlot(slot, data, dataEnd);
        break;
      case SlotDef::VECTOR_SPARSE_NON_VALUE:
        fillSparseNonValueSlot(slot, data, dataEnd);
        break;
      case SlotDef::VECTOR_SPARSE_VALUE:
        fillSparseValueSlot(slot, data, dataEnd);
        break;
      case SlotDef::INDEX:
        fillIndexSlot(slot, data, dataEnd);
        break;
      case SlotDef::VAR_MDIM_DENSE:
        LOG(FATAL) << "Not implemented";
        break;
      case SlotDef::VAR_MDIM_INDEX:
        LOG(FATAL) << "Not implemented";
        break;
      case SlotDef::STRING:
        fillStringSlot(slot, data, dataEnd);
        break;
    }
  }
  // read sequenceStartPositions
  for (size_t j = 0; j < slotNum_; ++j) {
    auto& slot = slots_[j];
    if (!iidData()) {
      unsigned int sequenceNum = readT<unsigned int>(data, dataEnd);
      slot.sequenceNum = sequenceNum;
      for (size_t i = 0; i < sequenceNum; ++i) {
        slot.sequenceStartPositions.push_back(
            readT<unsigned int>(data, dataEnd));
      }
      for (size_t i = 0; i < sequenceNum; ++i) {
        size_t begin = slot.sequenceStartPositions[i];
        size_t end = (i < sequenceNum - 1) ? slot.sequenceStartPositions[i + 1]
                                           : slot.sampleNum;
        for (size_t ii = begin; ii < end; ++ii) {
          slot.sampleSequenceIdVec.push_back(ii);
        }
      }
    } else {
      for (size_t i = 0; i < slot.sampleNum; ++i) {
        slot.sampleSequenceIdVec.push_back(i);
      }
    }
  }
  // read subSequenceStartPositions, not all slots have this infomation.
  for (size_t j = 0; j < slotNum_; ++j) {
    auto& slot = slots_[j];
    if (!iidData() && data != dataEnd) {
      unsigned int subSequenceNum = readT<unsigned int>(data, dataEnd);
      slot.subSequenceNum = subSequenceNum;
      for (size_t i = 0; i < subSequenceNum; ++i) {
        slot.subSequenceStartPositions.push_back(
            readT<unsigned int>(data, dataEnd));
      }
    }
  }
}

void PyDataProvider::reset() {
  {  // Invoke PyDataProvider Reset
    PyGuard guard;
    PyObjectPtr obj(PyObject_CallMethod(
        classInstance_.get(), const_cast<char*>("reset"), NULL));
    CHECK_PY(obj) << "Call function reset failed.";
  }

  if (!skipShuffle_) {
    // Invoke PyDataProvider Shuffle
    shuffle();
  }
  DataProvider::reset();
}

void PyDataProvider::shuffle() {
  // py shuffle
  PyGuard guard;
  PyObjectPtr obj(PyObject_CallMethod(
      classInstance_.get(), const_cast<char*>("shuffle"), NULL));
  CHECK_PY(obj) << "Call function shuffle failed.";
}

void PyDataProvider::handleDenseSlot(ProtoSlot& slot,
                                     size_t slotIndex,
                                     std::vector<Argument>& cpuArguments) {
  unsigned int dim = slot.dim;
  Matrix::resizeOrCreate(cpuArguments[slotIndex].value,
                         slot.sampleNum,
                         dim,
                         false,   // trans = false
                         false);  // useGpu = false
  real* buf = cpuArguments[slotIndex].value->getData();
  for (size_t i = 0; i < slot.sampleNum; ++i) {
    memcpyWithCheck(buf + i * dim,
                    slot.denseData.data() + slot.sampleSequenceIdVec[i] * dim,
                    sizeof(real) * dim,
                    slot.denseData.data() + slot.denseData.size());
  }
}

void PyDataProvider::handleSparseNonValueSlot(
    ProtoSlot& slot, size_t slotIndex, std::vector<Argument>& cpuArguments) {
  unsigned int dim = slot.dim;
  if (!(cpuArguments[slotIndex].value)) {
    cpuArguments[slotIndex].value =
        Matrix::createSparseMatrix(slot.sampleNum,
                                   dim,
                                   slot.sampleNum /*DEFAULT_AVG_WIDTH = 1*/,
                                   NO_VALUE,
                                   SPARSE_CSR,
                                   false,
                                   useGpu_);
  }
  auto mat = cpuArguments[slotIndex].value;
  mat->resize(slot.sampleNum, dim, slot.sampleNum, NO_VALUE, SPARSE_CSR);
  if (std::dynamic_pointer_cast<GpuSparseMatrix>(mat)) {
    std::dynamic_pointer_cast<GpuSparseMatrix>(mat)->copyFrom(
        slot.sampleSequenceIdVec.data(),
        slot.indices.data(),
        slot.sparseNonValueData.data(),
        HPPL_STREAM_1);
  } else if (std::dynamic_pointer_cast<CpuSparseMatrix>(mat)) {
    std::dynamic_pointer_cast<CpuSparseMatrix>(mat)->copyFrom(
        slot.sampleSequenceIdVec.data(),
        slot.indices.data(),
        slot.sparseNonValueData.data());
  } else {
    LOG(FATAL) << "Not Supported";
  }
}

void PyDataProvider::handleSparseValueSlot(
    ProtoSlot& slot, size_t slotIndex, std::vector<Argument>& cpuArguments) {
  unsigned int dim = slot.dim;
  if (!(cpuArguments[slotIndex].value)) {
    cpuArguments[slotIndex].value =
        Matrix::createSparseMatrix(slot.sampleNum,
                                   dim,
                                   slot.sampleNum /*DEFAULT_AVG_WIDTH = 1*/,
                                   FLOAT_VALUE,
                                   SPARSE_CSR,
                                   false,
                                   useGpu_);
  }
  auto mat = cpuArguments[slotIndex].value;
  mat->resize(slot.sampleNum, dim, slot.sampleNum, FLOAT_VALUE, SPARSE_CSR);
  if (std::dynamic_pointer_cast<GpuSparseMatrix>(mat)) {
    std::dynamic_pointer_cast<GpuSparseMatrix>(mat)->copyFrom(
        slot.sampleSequenceIdVec.data(),
        slot.indices.data(),
        slot.sparseFloatValueData.data(),
        HPPL_STREAM_DEFAULT);
  } else if (std::dynamic_pointer_cast<CpuSparseMatrix>(mat)) {
    std::dynamic_pointer_cast<CpuSparseMatrix>(mat)->copyFrom(
        slot.sampleSequenceIdVec.data(),
        slot.indices.data(),
        slot.sparseFloatValueData.data());
  } else {
    LOG(FATAL) << "Not Supported";
  }
}

void PyDataProvider::handleIndexSlot(ProtoSlot& slot,
                                     size_t slotIndex,
                                     std::vector<Argument>& cpuArguments) {
  IVector::resizeOrCreate(cpuArguments[slotIndex].ids,
                          slot.sampleNum,
                          /*useGpu_*/ false);
  int* buf = cpuArguments[slotIndex].ids->getData();
  for (size_t i = 0; i < slot.sampleNum; ++i) {
    buf[i] = slot.indexData[slot.sampleSequenceIdVec[i]];
  }
}

void PyDataProvider::handleStringSlot(ProtoSlot& slot,
                                      size_t slotIndex,
                                      std::vector<Argument>& cpuArguments) {
  if (cpuArguments[slotIndex].strs) {
    cpuArguments[slotIndex].strs->resize(slot.sampleNum);
  } else {
    cpuArguments[slotIndex].strs =
        std::make_shared<std::vector<std::string>>(slot.sampleNum);
  }
  for (size_t i = 0; i < slot.sampleNum; ++i) {
    (*cpuArguments[slotIndex].strs)[i] =
        slot.strData[slot.sampleSequenceIdVec[i]];
  }
}

int64_t PyDataProvider::getNextBatchInternal(int64_t size, DataBatch* batch) {
  PyGuard guard;
  PyObjectPtr obj(PyObject_CallMethod(classInstance_.get(),
                                      const_cast<char*>("getNextBatch"),
                                      const_cast<char*>("i"),
                                      size));
  CHECK_PY(obj) << "Call function getNextBatch failed.";
  const std::string& samples =
      std::string(PyString_AsString(obj.get()), PyString_Size(obj.get()));
  resetSlots();
  fillSlotsByStr(samples);
  size = batchSize_;
  if (size <= 0) return 0;

  DataBatch& cpuBatch = *cpuBatch_;
  std::vector<Argument>& cpuArguments = cpuBatch.getStreams();
  cpuBatch.setSize(size);
  cpuArguments.resize(slotNum_);

  if (!iidData()) {
    for (size_t j = 0; j < slotNum_; ++j) {
      auto& slot = slots_[j];
      ICpuGpuVector::resizeOrCreate(cpuArguments[j].sequenceStartPositions,
                                    slot.sequenceNum + 1,
                                    /* useGpu= */ false);
      int* buf = cpuArguments[j].sequenceStartPositions->getMutableData(false);
      std::copy(slot.sequenceStartPositions.begin(),
                slot.sequenceStartPositions.end(),
                buf);
      buf[slot.sequenceStartPositions.size()] = slot.sampleNum;

      if (slot.subSequenceStartPositions.size()) {
        ICpuGpuVector::resizeOrCreate(cpuArguments[j].subSequenceStartPositions,
                                      slot.subSequenceNum + 1,
                                      /*  useGpu= */ false);
        int* buf =
            cpuArguments[j].subSequenceStartPositions->getMutableData(false);
        std::copy(slot.subSequenceStartPositions.begin(),
                  slot.subSequenceStartPositions.end(),
                  buf);
        buf[slot.subSequenceNum] = slot.sampleNum;
        // check subSequenceStartPositions and sequenceStartPositions
        cpuArguments[j].checkSubset();
      }
    }
  }

  for (size_t slotIndex = 0; slotIndex < slotNum_; ++slotIndex) {
    auto& slot = slots_[slotIndex];
    SlotDef::SlotType slotType = slot.type;
    switch (slotType) {
      case SlotDef::VECTOR_DENSE:
        handleDenseSlot(slot, slotIndex, cpuArguments);
        break;
      case SlotDef::VECTOR_SPARSE_NON_VALUE:
        handleSparseNonValueSlot(slot, slotIndex, cpuArguments);
        break;
      case SlotDef::VECTOR_SPARSE_VALUE:
        handleSparseValueSlot(slot, slotIndex, cpuArguments);
        break;
      case SlotDef::INDEX:
        handleIndexSlot(slot, slotIndex, cpuArguments);
        break;
      case SlotDef::VAR_MDIM_DENSE:
        LOG(FATAL) << "Not implemented";
        break;
      case SlotDef::VAR_MDIM_INDEX:
        LOG(FATAL) << "Not implemented";
        break;
      case SlotDef::STRING:
        handleStringSlot(slot, slotIndex, cpuArguments);
        break;
    }
  }

  if (useGpu_) {
    std::vector<Argument>& cpuArguments = cpuBatch.getStreams();
    DataBatch& gpuBatch = *gpuBatch_;
    std::vector<Argument>& gpuArguments = gpuBatch.getStreams();
    gpuArguments.resize(cpuArguments.size());
    gpuBatch.setSize(size);
    for (size_t i = 0; i < slotNum_; ++i) {
      SlotDef::SlotType slotType = slots_[i].type;
      if (SlotDef::VECTOR_SPARSE_VALUE == slotType ||
          SlotDef::VECTOR_SPARSE_NON_VALUE == slotType) {
        gpuArguments[i] = cpuArguments[i];
        gpuArguments[i].sequenceStartPositions =
            cpuArguments[i].sequenceStartPositions;

        if (slots_[i].subSequenceStartPositions.size()) {
          gpuArguments[i].subSequenceStartPositions =
              cpuArguments[i].subSequenceStartPositions;
        }
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

  return batch->getSize();
}

}  // namespace paddle
