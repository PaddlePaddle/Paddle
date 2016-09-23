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

#include "paddle/utils/Stat.h"
#include "paddle/utils/StringUtil.h"
#include "ProtoReader.h"
#include "DataProvider.h"
#include "DataFormat.pb.h"


P_DECLARE_double(memory_threshold_on_load_data);

namespace paddle {
class ISlotScanner {
public:
  virtual ~ISlotScanner() {}
  virtual void startScan1(Argument& arg, int dim,
                          DataHeader2::SeqType seqType) = 0;
  // make some statistics, and malloc memory for variables
  virtual void scan1(Argument& arg, const SlotSample& sample) = 0;
  virtual void finishScan1(Argument& arg) = 0;

  virtual void startScan2(Argument& arg, int dim,
                                DataHeader2::SeqType seqType) = 0;
  // fill data
  virtual void scan2(Argument& arg, const SlotSample& sample) = 0;
  virtual void finishScan2(Argument& arg) = 0;

  static ISlotScanner* create(SlotDef slotDef, DataHeader2::SeqType seqType);
};

class ProtoDataProvider2 : public DataProvider {
public:
  ProtoDataProvider2(const DataConfig& config, bool useGpu,
                     bool loadDataAll = true);
  virtual void reset();

  virtual int64_t getSize() {
    int64_t size = sampleNums_;
    if (usageRatio_ < 1.0f) {
      size = static_cast<int64_t >(size * usageRatio_);
    }
    return size;
  }

  virtual void shuffle();
  void loadData(const std::vector<std::string>& fileList);

  virtual int64_t getNextBatchInternal(int64_t size, DataBatch* batch);

protected:
  void loadData(const std::string& fileName);
  void loadDataFile(const std::string& fileName);
  void initDataHeader(const DataHeader2& header);

private:
  void process(const std::vector<std::shared_ptr<DataSample2>>& samples,
               DataBatch* batch);

protected:
  DataHeader2 header_;
  std::vector<std::shared_ptr<DataSample2>> sample_;
  size_t sampleNums_;
  std::vector<std::unique_ptr<ISlotScanner>> scanners_;
  int64_t currentIndex_;
  std::mutex lock_;
  std::once_flag once_flag_;

  ThreadLocalD<DataBatch> cpuBatch_;
  ThreadLocalD<DataBatch> gpuBatch_;
};



ProtoDataProvider2::ProtoDataProvider2(const DataConfig& config, bool useGpu,
                                     bool loadDataAll)
        : DataProvider(config, useGpu), sampleNums_(0), currentIndex_(0) {
  if (loadDataAll) {
    loadData(config_.files());
  }
}

void ProtoDataProvider2::process(
  const std::vector<std::shared_ptr<DataSample2>>& samples,
  DataBatch* batch) {
  batch->setSize(samples.size());
  batch->getStreams().resize(this->header_.slot_defs_size());
#define ScanLoop(ID)                                                   \
do {                                                                   \
  for (size_t i = 0; i < this->scanners_.size(); ++i) {                \
    Argument& arg = batch->getStreams()[i];                            \
    ISlotScanner& scanner = *this->scanners_[i];                       \
    scanner.startScan##ID(arg, this->header_.slot_defs(i).dim(),       \
                          this->header_.seq_type(i));                  \
    for (const std::shared_ptr<DataSample2>& samplePtr : samples) {    \
      const DataSample2& sample = *samplePtr;                          \
      for (const SlotSample& dat : sample.slots_data()) {              \
        if (dat.slot_id() == i) {                                      \
          scanner.scan##ID(arg, dat);                                  \
          break;                                                       \
        }                                                              \
      }                                                                \
    }                                                                  \
    scanner.finishScan##ID(arg);                                       \
  }                                                                    \
} while (0)
  ScanLoop(1);
  ScanLoop(2);
}

void ProtoDataProvider2::loadData(const std::vector<std::string>& fileList) {
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
  LOG(INFO) << "read done, num of instance=" << sampleNums_;
}

void ProtoDataProvider2::loadData(const std::string& fileName) {
  std::vector<std::string> fileList;
  loadFileList(fileName, fileList);
  loadData(fileList);
}


void ProtoDataProvider2::initDataHeader(const DataHeader2& header) {
  std::call_once(once_flag_, [&] {
    header_ = header;
    this->scanners_.resize(header_.slot_defs_size());
    for (int i = 0; i < header_.slot_defs_size(); ++i) {
      this->scanners_[i].reset(
              ISlotScanner::create(header_.slot_defs(i), header_.seq_type(i)));
    }
  });
}

void ProtoDataProvider2::loadDataFile(const std::string& fileName) {
  std::ifstream is(fileName);
  CHECK(is) << "Fail to open " << fileName;
  bool dataCompression = str::endsWith(fileName, ".gz");
  std::unique_ptr<ProtoReader> reader(new ProtoReader(&is, dataCompression));
  CHECK(reader) << "Fail to create proto data input stream";

  DataHeader2 header;
  CHECK(reader->read(&header));
  initDataHeader(header);


  while (true) {
    DataSample2* samplePtr = new DataSample2();
    if (!reader->read(samplePtr)) {
      break;
    }
    sample_.push_back(std::shared_ptr<DataSample2>(samplePtr));
    ++sampleNums_;
  }

  CHECK(is.eof()) << "Fail to read file";
  reader.reset(nullptr);
  is.close();
}


int64_t ProtoDataProvider2::getNextBatchInternal(int64_t size,
                                                 DataBatch *batchOut) {
  int64_t actualSize = 0;
  std::lock_guard<std::mutex> guard(lock_);
  std::vector<std::shared_ptr<DataSample2>> buf;
  buf.reserve(size);
  for (; actualSize < size &&
         currentIndex_ + actualSize < static_cast<int64_t>(sampleNums_);
          ++actualSize) {
    buf.push_back(sample_[currentIndex_ + actualSize]);
  }
  currentIndex_ += actualSize;
  if (actualSize > 0) {
    DataBatch& cpuBatch = *cpuBatch_;
    process(buf, &cpuBatch);
    if (useGpu_) {
      std::vector<Argument>& cpuArgs = cpuBatch.getStreams();
      DataBatch& gpuBatch = *gpuBatch_;
      std::vector<Argument>& gpuArgs = gpuBatch.getStreams();
      gpuArgs.resize(cpuArgs.size());
      gpuBatch.setSize(size);
      for (size_t i = 0; i < cpuArgs.size(); ++i) {
        gpuArgs[i].resizeAndCopyFrom(cpuArgs[i], useGpu_, HPPL_STREAM_1);
      }
      hl_stream_synchronize(HPPL_STREAM_1);
      *batchOut = gpuBatch;
    } else {
      *batchOut = cpuBatch;
    }
  }
  return actualSize;
}

void ProtoDataProvider2::shuffle() {
  std::shuffle(sample_.begin(), sample_.end(), ThreadLocalRandomEngine::get());
}

void ProtoDataProvider2::reset() {
  currentIndex_ = 0;
  if (!skipShuffle_) {
    shuffle();
  }
  DataProvider::reset();
}

REGISTER_DATA_PROVIDER(proto2, ProtoDataProvider2);
// A DataSample have some slotSamples
// CompositeScanner is a vector of slotScanner
// One slotScanner for one slotSample
class CompositeScanner : public ISlotScanner {
public:
  void appendScanner(std::unique_ptr<ISlotScanner>&& ptr) {
    scanners_.push_back(std::move(ptr));
  }

private:
  std::list<std::unique_ptr<ISlotScanner>> scanners_;
// ISlotScanner interface
public:
  void startScan1(Argument& arg, int dim, DataHeader2::SeqType seqType) {
    for (std::unique_ptr<ISlotScanner>& scanner : scanners_) {
      scanner->startScan1(arg, dim, seqType);
    }
  }

  void scan1(Argument& arg, const SlotSample& sample) {
    for (std::unique_ptr<ISlotScanner>& scanner : scanners_) {
      scanner->scan1(arg, sample);
    }
  }

  void finishScan1(Argument& arg) {
    for (std::unique_ptr<ISlotScanner>& scanner : scanners_) {
      scanner->finishScan1(arg);
    }
  }

  void startScan2(Argument& arg, int dim, DataHeader2::SeqType seqType) {
    for (std::unique_ptr<ISlotScanner>& scanner : scanners_) {
      scanner->startScan2(arg, dim, seqType);
    }
  }

  void scan2(Argument& arg, const SlotSample& sample) {
    for (std::unique_ptr<ISlotScanner>& scanner : scanners_) {
      scanner->scan2(arg, sample);
    }
  }

  void finishScan2(Argument& arg) {
    for (std::unique_ptr<ISlotScanner>& scanner : scanners_) {
      scanner->finishScan2(arg);
    }
  }
};
// slotScanner for dense data
class ProtoDenseScanner : public ISlotScanner {
private:
  size_t rowCount_;
  size_t dim_;

public:
  void startScan1(Argument&, int dim, DataHeader2::SeqType seqType) {
    dim_ = dim;
    rowCount_ = 0;
  }

  void scan1(Argument&, const SlotSample& sample) {
    rowCount_ += sample.vector_slots_size();
  }

  void finishScan1(Argument& arg) {
    Matrix::resizeOrCreate(arg.value, rowCount_, dim_, false, false);
  }

  void startScan2(Argument&, int, DataHeader2::SeqType seqType) {
    rowCount_ = 0;
  }
  void scan2(Argument& arg, const SlotSample& sample) {
    for (auto& vectorData : sample.vector_slots()) {
      auto& values = vectorData.values();
      auto dataBegin = arg.value->getData() + dim_ * rowCount_;
      std::copy(values.begin(), values.end(), dataBegin);
      ++rowCount_;
    }
  }

  void finishScan2(Argument& arg) {
    // Do nothing
  }
};
// slotScanner for index data
class ProtoIndexScanner : public ISlotScanner {
private:
  size_t indexCount_;
  size_t offset_;

public:
  void startScan1(Argument& arg, int dim, DataHeader2::SeqType seqType) {
    indexCount_ = 0;
  }

  void scan1(Argument& arg, const SlotSample& sample) {
    indexCount_ += sample.vector_slots_size();
  }

  void finishScan1(Argument& arg) {
    IVector::resizeOrCreate(arg.ids, indexCount_, false);
  }

  void startScan2(Argument& arg, int dim, DataHeader2::SeqType seqType) {
     offset_ = 0;
  }

  void scan2(Argument& arg, const SlotSample& sample) {
    for (auto& vectorData : sample.vector_slots()) {
      arg.ids->getData()[offset_++] = vectorData.ids(0);
    }
  }

  void finishScan2(Argument& arg) {
    // Do nothing
  }
};

class ProtoSparseNonValueScanner : public ISlotScanner {
protected:
  size_t nnz_;
  size_t width_;
  size_t height_;

  int* col_;
  int* row_;

  size_t colOffset_;
  size_t rowOffset_;

public:
  void startScan1(Argument& arg, int dim, DataHeader2::SeqType seqType) {
    nnz_ = 0;
    width_ = dim;
    height_ = 0;
  }

  void scan1(Argument& arg, const SlotSample& sample) {
    for (auto& vectorData : sample.vector_slots()) {
      nnz_ += vectorData.ids_size();
      ++height_;
    }
  }

  void finishScan1(Argument& arg) {
    Matrix::resizeOrCreateSparseMatrix(arg.value, height_, width_, nnz_,
                                       NO_VALUE);
  }

  void startScan2(Argument& arg, int dim, DataHeader2::SeqType seqType) {
    auto smat = dynamic_cast<CpuSparseMatrix*>(arg.value.get());
    col_ = smat->getCols();
    row_ = smat->getRows();
    row_[0] = 0;
    colOffset_ = 0;
    rowOffset_ = 1;
  }

  void scan2(Argument& arg, const SlotSample& sample) {
    for (auto& vectorData : sample.vector_slots()) {
      row_[rowOffset_] = vectorData.ids_size() + row_[rowOffset_ - 1];
      ++rowOffset_;
      std::copy(vectorData.ids().begin(), vectorData.ids().end(),
                col_ + colOffset_);
      colOffset_ += vectorData.ids_size();
    }
  }

  void finishScan2(Argument& arg) {
    // Do nothing
  }
};

class ProtoSparseValueScanner : public ProtoSparseNonValueScanner {
private:
  float* value_;

public:
  void finishScan1(Argument& arg) {
    Matrix::resizeOrCreateSparseMatrix(arg.value, height_, width_, nnz_,
                                       FLOAT_VALUE);
  }

  void startScan2(Argument& arg, int dim, DataHeader2::SeqType seqType) {
    ProtoSparseNonValueScanner::startScan2(arg, dim, seqType);
    value_ =
      dynamic_cast<paddle::CpuSparseMatrix*>(arg.value.get())->getValue();
  }
  void scan2(Argument& arg, const SlotSample& sample) {
    for (auto& vectorData : sample.vector_slots()) {
      row_[rowOffset_] = vectorData.ids_size() + row_[rowOffset_ - 1];
      ++rowOffset_;
      std::copy(vectorData.ids().begin(), vectorData.ids().end(),
                col_ + colOffset_);
      std::copy(vectorData.values().begin(), vectorData.values().end(),
                value_ + colOffset_);
      colOffset_ += vectorData.ids_size();
    }
  }
};
// slotScanner for sequence data
class ProtoSequenceSlotScanner : public ISlotScanner {
private:
  size_t seqCount_;
  size_t seqOffset_;
  size_t seqBeginningPos_;

public:
  void startScan1(Argument& arg, int dim, DataHeader2::SeqType seqType) {
    seqCount_ = 0;
    seqOffset_ = 0;
  }

  void scan1(Argument& arg, const SlotSample& sample) {
    ++seqCount_;
  }

  void finishScan1(Argument& arg) {
    ICpuGpuVector::resizeOrCreate(arg.sequenceStartPositions, seqCount_ + 1,
                              false);
  }

  void startScan2(Argument& arg, int dim, DataHeader2::SeqType seqType) {
    seqBeginningPos_ = 0;
  }

  void scan2(Argument& arg, const SlotSample& sample) {
    arg.sequenceStartPositions->getMutableData(false)[seqOffset_++] =
                                                      seqBeginningPos_;
    seqBeginningPos_ += sample.vector_slots_size();
  }

  void finishScan2(Argument& arg) {
    arg.sequenceStartPositions->getMutableData(false)[seqOffset_] =
                                                    seqBeginningPos_;
  }
};

// slotScanner for subsequence data
class ProtoSubSequenceSlotScanner : public ISlotScanner {
private:
  size_t subSeqCount_;
  size_t subSeqOffset_;
  size_t subSeqBeginningPos_;

public:
  void startScan1(Argument& arg, int dim, DataHeader2::SeqType seqType) {
    subSeqCount_ = 0;
  }

  void scan1(Argument& arg, const SlotSample& sample) {
    subSeqCount_ += sample.subseq_start_positions_size();
  }

  void finishScan1(Argument& arg) {
    ICpuGpuVector::resizeOrCreate(arg.subSequenceStartPositions,
                            subSeqCount_ + 1, false);
  }

  void startScan2(Argument& arg, int dim, DataHeader2::SeqType seqType) {
    subSeqOffset_ = 0;
    subSeqBeginningPos_ = 0;
  }

  void scan2(Argument& arg, const SlotSample& sample) {
    auto startPos = arg.subSequenceStartPositions->getMutableData(false);
    for (auto& subSeqStartPositions : sample.subseq_start_positions()) {
      startPos[subSeqOffset_++] = subSeqBeginningPos_ + subSeqStartPositions;
    }
    subSeqBeginningPos_ += sample.vector_slots_size();
  }

  void finishScan2(Argument& arg) {
    arg.subSequenceStartPositions->getMutableData(false)[subSeqOffset_] =
                                                      subSeqBeginningPos_;
  }
};

ISlotScanner* ISlotScanner::create(SlotDef slotDef,
                           DataHeader2::SeqType seqType) {
CompositeScanner* scanner = new CompositeScanner();
  switch (slotDef.type()) {
    case SlotDef::VECTOR_DENSE:
      scanner->appendScanner(
          std::unique_ptr<ISlotScanner>(new ProtoDenseScanner));
      break;
    case SlotDef::INDEX:
      scanner->appendScanner(
          std::unique_ptr<ISlotScanner>(new ProtoIndexScanner));
      break;
    case SlotDef::VECTOR_SPARSE_NON_VALUE:
      scanner->appendScanner(
              std::unique_ptr<ISlotScanner>(new ProtoSparseNonValueScanner));
      break;
    case SlotDef::VECTOR_SPARSE_VALUE:
      scanner->appendScanner(
              std::unique_ptr<ISlotScanner>(new ProtoSparseValueScanner));
      break;
    default:
      LOG(FATAL) << slotDef.type() << " is not implemented";
      break;
  }

  switch (seqType) {
    case DataHeader2::NON_SEQ:
      //! Do nothing.
      break;
    case DataHeader2::SEQ:
      scanner->appendScanner(
          std::unique_ptr<ISlotScanner>(new ProtoSequenceSlotScanner));
      break;
    case DataHeader2::SUB_SEQ:
      scanner->appendScanner(
              std::unique_ptr<ISlotScanner>(new ProtoSequenceSlotScanner));
      scanner->appendScanner(
          std::unique_ptr<ISlotScanner>(new ProtoSubSequenceSlotScanner));
      break;
    default:
      LOG(FATAL) << "Sequence type " << seqType <<
      " is not implemented";
          break;
  }
  return scanner;
}

}  // namespace paddle
