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

class MultiDataProvider : public DataProvider {
 protected:
  std::vector<std::unique_ptr<DataProvider>> subDataProviders_;

 public:
  MultiDataProvider(const DataConfig& config,
                    const ModelConfig& modelConfig,
                    bool useGpu);
  ~MultiDataProvider() {}
  virtual void reset();
  virtual void shuffle();
  virtual int64_t getSize() { return -1; }
  virtual int64_t getNextBatchInternal(int64_t size, DataBatch* batch);
  bool isTestMode() const { return isTestMode_; }

 private:
  int totalDataRatio_;
  bool isTestMode_;
};

}  // namespace paddle
