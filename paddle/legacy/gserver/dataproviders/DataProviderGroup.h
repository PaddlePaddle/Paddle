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

#include "DataProvider.h"

namespace paddle {

template <class T>
class DataProviderGroup : public DataProvider {
 protected:
  typedef T ProviderType;
  typedef std::shared_ptr<ProviderType> ProviderPtrType;
  ProviderPtrType provider_;

  std::vector<std::string> fileList_;
  std::mutex lock_;
  std::unique_ptr<MultiThreadWorker<ProviderType>> loader_;

 public:
  DataProviderGroup(const DataConfig& config, bool useGpu);
  ~DataProviderGroup() {}

  virtual void reset();
  virtual void shuffle() {}
  virtual int64_t getSize() { return -1; }
  virtual int64_t getNextBatchInternal(int64_t size, DataBatch* batch);

 private:
  void startLoader();
  void stopLoader();
  void forceStopLoader();
  ProviderPtrType loadFile(const std::vector<std::string>& fileList);
};

template <class T>
DataProviderGroup<T>::DataProviderGroup(const DataConfig& config, bool useGpu)
    : DataProvider(config, useGpu) {
  // load file list
  loadFileList(config_.files(), fileList_);
  CHECK_GT(fileList_.size(), 0LU);
  LOG(INFO) << "load file list, numfiles=" << fileList_.size()
            << ", max_num_of_data_providers_in_memory="
            << (1 + config_.file_group_conf().queue_capacity() +
                config_.file_group_conf().load_thread_num());
}

template <class T>
void DataProviderGroup<T>::reset() {
  forceStopLoader();
  CHECK(!loader_);
  provider_ = nullptr;

  // shuffle file list
  std::shuffle(
      fileList_.begin(), fileList_.end(), ThreadLocalRandomEngine::get());

  startLoader();
  DataProvider::reset();
}

template <class T>
int64_t DataProviderGroup<T>::getNextBatchInternal(int64_t size,
                                                   DataBatch* batch) {
  std::lock_guard<std::mutex> guard(lock_);

  if (!loader_) {
    return 0;
  }
  if (provider_) {
    int64_t ret = provider_->getNextBatchInternal(size, batch);
    if (ret > 0) {
      return ret;
    }
  }

  // else get data from next data provider
  if (loader_->testResult()) {
    LOG(INFO) << "WAIT provider";
  }
  provider_ = loader_->waitResult();
  if (!provider_) {
    stopLoader();  // All the data providers have been returned
    return 0;
  }
  int64_t ret = provider_->getNextBatchInternal(size, batch);
  CHECK(ret > 0) << "new data provider does not contain any valid samples!";
  return ret;
}

template <class T>
void DataProviderGroup<T>::startLoader() {
  loader_.reset(new MultiThreadWorker<ProviderType>(
      config_.file_group_conf().load_thread_num(),
      config_.file_group_conf().queue_capacity()));

  int loadFileCount = config_.file_group_conf().load_file_count();
  for (size_t startPos = 0; startPos < fileList_.size();
       startPos += loadFileCount) {
    size_t endPos = std::min(fileList_.size(), startPos + loadFileCount);
    std::vector<std::string> fileVec(fileList_.begin() + startPos,
                                     fileList_.begin() + endPos);
    loader_->addJob([this, fileVec]() -> ProviderPtrType {
      return this->loadFile(fileVec);
    });
  }
  loader_->stopAddJob();
}

template <class T>
void DataProviderGroup<T>::stopLoader() {
  if (loader_) {
    loader_->stop();
    loader_ = nullptr;
  }
}

template <class T>
void DataProviderGroup<T>::forceStopLoader() {
  if (loader_) {
    loader_->forceStop();
    loader_ = nullptr;
  }
}

template <class T>
std::shared_ptr<T> DataProviderGroup<T>::loadFile(
    const std::vector<std::string>& fileList) {
  // disable async_load_data in sub dataprovider
  DataConfig subConfig = config_;
  subConfig.set_async_load_data(false);

  CHECK(!fileList.empty()) << "fileList is empty";
  ProviderPtrType provider =
      std::make_shared<ProviderType>(subConfig, useGpu_, false);
  provider->loadData(fileList);
  provider->reset();
  return provider;
}

}  // namespace paddle
