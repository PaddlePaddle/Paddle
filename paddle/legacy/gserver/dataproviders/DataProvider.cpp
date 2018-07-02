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

#include "DataProvider.h"

#include <unistd.h>
#include <algorithm>
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/StringUtil.h"
#include "paddle/utils/Util.h"

namespace paddle {

void BufferBatch::swap(BufferBatch* bufBatch) {
  DataBatch* batchData = bufBatch->getDataBatch();
  hl_event_t hlEvent = bufBatch->getCuEvent();
  hl_stream_t hlStream = bufBatch->getCuStream();
  bufBatch->setDataBatch(batchData_);
  bufBatch->setCuStream(hlStream_);
  bufBatch->setCuEvent(hlEvent_);

  batchData_ = batchData;
  hlEvent_ = hlEvent;
  hlStream_ = hlStream;
}

void BufferBatch::clone(DataBatch* srcBatch, bool useGpu) {
  if (batchData_ == NULL) {
    batchData_ = new DataBatch();
  }
  std::vector<Argument>& destData = batchData_->getStreams();
  int numStreams = srcBatch->getNumStreams();
  destData.resize(numStreams);
  batchData_->setSize(srcBatch->getSize());
  if (useGpu) {
    createCuEvent();
  }

  for (int i = 0; i < numStreams; i++) {
    destData[i].resizeAndCopyFrom(srcBatch->getStream(i), useGpu, hlStream_);
  }
  if (useGpu) {
    hl_stream_record_event(hlStream_, hlEvent_);
  }
}

DoubleBuffer::DoubleBuffer(DataProvider* dataPool,
                           bool useGpu,
                           int64_t batchSize) {
  batchSize_ = batchSize;
  dataPool_ = dataPool;
  useGpu_ = useGpu;
  dataQueue_ = new BufferBatchQueue();
  bufferQueue_ = new BufferBatchQueue();

  // insert a empty buffer
  bufferQueue_->enqueue(new BufferBatch());
  stopping_ = false;
  pending_ = true;
}

DoubleBuffer::~DoubleBuffer() {
  finishAsyncLoad();
  while (dataQueue_->size()) {
    BufferBatch* dataBtch = dataQueue_->dequeue();
    delete dataBtch;
    dataBtch = NULL;
  }
  while (bufferQueue_->size()) {
    BufferBatch* bufBtch = bufferQueue_->dequeue();
    delete bufBtch;
    bufBtch = NULL;
  }
  delete dataQueue_;
  dataQueue_ = NULL;
  delete bufferQueue_;
  bufferQueue_ = NULL;
}

void DoubleBuffer::removeOneBatch(DataBatch* dataBatch) {
  // get data
  BufferBatch* batch = dataQueue_->dequeue();
  batch->syncEvent();  // when use GPU, need synchronized with the cuEvent
  *dataBatch = *(batch->getDataBatch());

  // push anothor buffer
  if (*usingBatch_ == nullptr) {
    *usingBatch_ = std::make_shared<BufferBatch>();
  }

  // Mark the using-batch
  batch->swap((*usingBatch_).get());
  bufferQueue_->enqueue(batch);

  if (0 == dataBatch->getSize()) {
    setPending(true);
  }
}

void DoubleBuffer::insertOneBatch(DataBatch* batch) {
  while (!bufferQueue_->waitNotEmptyFor(2 /* seconds */)) {  // time out
    if (stopping_) return;
  }
  BufferBatch* bufBatch = bufferQueue_->dequeue();
  // clone and copy the data from an Threadlocal Variable
  bufBatch->clone(batch, useGpu_);
  dataQueue_->enqueue(bufBatch);
}

void DoubleBuffer::asyncLoadBatch() {
  int64_t actualSize = 0;
  if (useGpu_) {
    hl_set_device(FLAGS_gpu_id);
  }
  setPending(false);

  while (true) {
    taskReadySem_.wait();
    if (stopping_) break;

    while (batchSize_ == 0 && !stopping_) {
      usleep(5);
    }
    if (stopping_) break;

    do {
      DataBatch newBatch;
      {
        REGISTER_TIMER("getNextBatchInternal");
        actualSize = dataPool_->getNextBatchInternal(batchSize_, &newBatch);
      }
      insertOneBatch(&newBatch);
    } while (actualSize > 0 && !stopping_);
  }
}

void DoubleBuffer::startAsyncLoad() {
  if (asyncLoader_ == nullptr) {
    asyncLoader_.reset(new std::thread([this]() { this->asyncLoadBatch(); }));
  }
  taskReadySem_.post();
}

ClassRegistrar<DataProvider, DataConfig, ModelConfig, bool>
    DataProvider::registrar_;

DataProvider* DataProvider::create(const DataConfig& config,
                                   const ModelConfig& modelConfig,
                                   bool useGpu) {
  return registrar_.createByType(config.type(), config, modelConfig, useGpu);
}

REGISTER_DATA_PROVIDER(simple, SimpleDataProvider);
REGISTER_DATA_PROVIDER(dummy, DummyDataProvider);

int64_t DataProvider::getNextBatch(int64_t size, DataBatch* batch) {
  int64_t batchSize = doubleBuffer_ ? getNextBatchFromBuffer(size, batch)
                                    : getNextBatchInternal(size, batch);

  if (!batchSize) return 0;

  if (!config_.constant_slots_size()) return batchSize;

  auto& constantSlots = *constantSlots_;
  constantSlots.resize(config_.constant_slots_size());

  for (int i = 0; i < config_.constant_slots_size(); ++i) {
    MemoryHandlePtr handle =
        constantSlots[i] ? constantSlots[i]->getMemoryHandle() : nullptr;
    Matrix::resizeOrCreate(constantSlots[i],
                           batchSize,
                           1,         // = width
                           false,     // = trans
                           useGpu_);  // = useGpu
    if (handle != constantSlots[i]->getMemoryHandle()) {
      // memory buf was reallocated. We need to initialize the value
      constantSlots[i]->assign(config_.constant_slots(i));
    }
    batch->appendData(constantSlots[i],
                      batch->getStream(0).sequenceStartPositions);
  }

  return batchSize;
}

int64_t DataProvider::getNextBatchFromBuffer(int64_t size, DataBatch* batch) {
  CHECK(doubleBuffer_ != nullptr);

  if (doubleBuffer_->getBatchSize() != size) {
    doubleBuffer_->setBatchSize(size);
  }

  doubleBuffer_->removeOneBatch(batch);
  return batch->getSize();
}

void DataProvider::initAsyncLoader() {
  if (doubleBuffer_ == nullptr) {
    doubleBuffer_.reset(new DoubleBuffer(this, useGpu_));
  }
  useGpu_ = false;  // Avoid D2D copy, it will delay the computing performance
}

SimpleDataProviderBase::SimpleDataProviderBase(const DataConfig& config,
                                               bool useGpu,
                                               bool withInfo)
    : DataProvider(config, useGpu) {
  /* initialize the size of a sample, and the buffer */
  sampleDim_ = config_.feat_dim() * (2 * config_.context_len() + 1);
  bufferCapacity_ = config_.buffer_capacity();
  withInfo_ = withInfo;
  sampleNumInBuf_ = 0;
  nextItemIndex_ = 0;

  /* malloc buffer in cpu */
  hInputDataBuf_ = std::make_shared<CpuMatrix>(bufferCapacity_, sampleDim_);
  hInputLabelBuf_ = std::make_shared<CpuIVector>(bufferCapacity_);
  hInputInfoBuf_ = std::make_shared<CpuIVector>(bufferCapacity_);
}

void SimpleDataProviderBase::shuffle() {
  int i, t;
  int len = sampleNumInBuf_;
  std::vector<real> temp(sampleDim_);
  real* data = hInputDataBuf_->getData();
  int* label = hInputLabelBuf_->getData();
  int* info = hInputInfoBuf_->getData();
  int sampleSz = sizeof(real) * sampleDim_;
  for (i = 0; i < len; i++) {
    int randNum = rand();  // NOLINT TODO(yuyang18): Use rand_r instead?
    t = randNum % (len - i) + i;
    // swap
    if (i != t) {
      // swap data
      memcpy(&temp[0], &data[i * sampleDim_], sampleSz);
      memcpy(&data[i * sampleDim_], &data[t * sampleDim_], sampleSz);
      memcpy(&data[t * sampleDim_], &temp[0], sampleSz);
      std::swap(label[i], label[t]);
      if (withInfo_) {
        std::swap(info[i], info[t]);
      }
    }
  }
}

int64_t SimpleDataProviderBase::getNextBatchInternal(int64_t size,
                                                     DataBatch* batch) {
  CHECK(batch != NULL);
  batch->clear();

  int64_t startIndex;
  int64_t cpySize;

  std::lock_guard<RWLock> guard(lock_);
  if (sampleNumInBuf_ - nextItemIndex_ < size) {
    int64_t n = fillBuffer();
    VLOG(1) << "fillBuffer return " << n << " samples.\n";
  }

  startIndex = nextItemIndex_;
  cpySize = std::min(size, sampleNumInBuf_ - nextItemIndex_);
  nextItemIndex_ += cpySize;

  if (cpySize > 0) {
    real* data = hInputDataBuf_->getData() + startIndex * sampleDim_;
    int* label = hInputLabelBuf_->getData() + startIndex;
    int* info = hInputInfoBuf_->getData() + startIndex;

    MatrixPtr& dataBatch = *dataBatch_;     // get the thread local object
    IVectorPtr& labelBatch = *labelBatch_;  // get the thread local object
    IVectorPtr& infoBatch = *infoBatch_;    // get the thread local object
    if (!dataBatch) {
      dataBatch = Matrix::create(cpySize, sampleDim_, false, useGpu_);
      labelBatch = IVector::create(cpySize, useGpu_);
      if (withInfo_) {
        infoBatch = IVector::create(cpySize, 0);
      }
    } else {
      dataBatch->resize(cpySize, sampleDim_);
      labelBatch->resize(cpySize);
      if (withInfo_) {
        infoBatch->resize(cpySize);
      }
    }
    dataBatch->copyFrom(data, cpySize * sampleDim_);
    labelBatch->copyFrom(label, cpySize);
    batch->appendData(dataBatch);
    batch->appendLabel(labelBatch);
    if (withInfo_) {
      infoBatch->copyFrom(info, cpySize);
      batch->appendLabel(infoBatch);
    }
  }

  batch->setSize(cpySize);
  return cpySize;
}

void SimpleDataProviderBase::reset() {
  sampleNumInBuf_ = 0;
  nextItemIndex_ = 0;
  DataProvider::reset();
}

int64_t SimpleDataProviderBase::getSize() {
  LOG(FATAL) << "Currently, not implemented";
  return 0;
}

int64_t SimpleDataProviderBase::fillBuffer() {
  int64_t n = sampleNumInBuf_ - nextItemIndex_;

  /* flash the remaining data to the beginning of the buffer */
  if (n > 0) {
    hInputDataBuf_->copyFrom(
        hInputDataBuf_->getData() + nextItemIndex_ * sampleDim_,
        n * sampleDim_);
    hInputLabelBuf_->copyFrom(hInputLabelBuf_->getData() + nextItemIndex_, n);
    if (withInfo_) {
      hInputInfoBuf_->copyFrom(hInputInfoBuf_->getData() + nextItemIndex_, n);
    }
  }

  sampleNumInBuf_ =
      n + fillBufferImp(hInputDataBuf_->getData() + n * sampleDim_,
                        hInputLabelBuf_->getData() + n,
                        hInputInfoBuf_->getData() + n,
                        bufferCapacity_ - n);

  /* for stachastic gradient training */
  if (!skipShuffle_) {
    shuffle();
  }

  nextItemIndex_ = 0;

  return sampleNumInBuf_;
}

SimpleDataProvider::SimpleDataProvider(const DataConfig& config, bool useGpu)
    : SimpleDataProviderBase(config, useGpu, /* withInfo= */ false),
      currentSampleIndex_(0) {
  loadData(config_.files());
}

SimpleDataProvider::~SimpleDataProvider() {}

int64_t SimpleDataProvider::fillBufferImp(real* data,
                                          int* label,
                                          int* info,
                                          int64_t size) {
  (void)info;
  int64_t n = std::min<int64_t>(labels_.size() - currentSampleIndex_, size);
  memcpy(data,
         &data_[currentSampleIndex_ * sampleDim_],
         n * sampleDim_ * sizeof(real));
  memcpy(label, &labels_[currentSampleIndex_], sizeof(int) * n);
  currentSampleIndex_ += n;

  return n;
}

void SimpleDataProvider::reset() {
  currentSampleIndex_ = 0;
  SimpleDataProviderBase::reset();
}

void SimpleDataProvider::loadData(const std::string& fileName) {
  std::ifstream is(fileName);
  CHECK(is) << "Fail to open " << fileName;
  std::string line;
  while (is) {
    if (!getline(is, line)) break;
    LOG(INFO) << "load data file " << line;
    loadDataFile(line);
  }
  LOG(INFO) << "read done, num of instance=" << labels_.size()
            << " data size=" << data_.size();
}

void SimpleDataProvider::loadDataFile(const std::string& fileName) {
  std::ifstream is(fileName);
  std::string line;
  std::vector<std::string> pieces;
  while (is) {
    if (!getline(is, line)) break;
    str::split(line, ' ', &pieces);
    CHECK_EQ((uint64_t)(sampleDim_ + 1), pieces.size())
        << " Dimension mismatch, " << pieces.size() - 1 << " in " << fileName
        << " " << sampleDim_ << " from config";
    labels_.push_back(atoi(pieces[0].c_str()));
    for (int i = 0; i < sampleDim_; ++i) {
      data_.push_back(atof(pieces[i + 1].c_str()));
    }
  }
}

}  // namespace paddle
