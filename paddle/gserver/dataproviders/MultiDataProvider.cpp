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

#include "MultiDataProvider.h"
#include <algorithm>
#include "paddle/utils/Logging.h"
#include "paddle/utils/Util.h"

namespace paddle {

using namespace std;

MultiDataProvider::MultiDataProvider(const DataConfig& config,
                                     const ModelConfig& modelConfig,
                                     bool useGpu)
    : DataProvider(config, useGpu) {
  bool atLeastOneMainDataFlag = false;
  totalDataRatio_ = 0;
  LOG(INFO) << "MultiDataProvider: sub data provider size: "
            << config.sub_data_configs_size();
  LOG(INFO) << "MultiDataProvider: for_test: " << config.for_test();
  isTestMode_ = config.for_test();
  for (int i = 0; i < config.sub_data_configs_size(); i++) {
    LOG(INFO) << "dataRatio of sub(" << i
              << ") is: " << config.sub_data_configs(i).data_ratio();
    totalDataRatio_ += config.sub_data_configs(i).data_ratio();
    if (config.sub_data_configs(i).is_main_data()) {
      LOG(INFO) << "main data is [" << i << "]";
      atLeastOneMainDataFlag = true;
    }
  }
  CHECK(atLeastOneMainDataFlag) << "all sub dataproviders in MultiData do not"
                                << " have is_main_data flag";
  LOG(INFO) << "totalDataRatio_=" << totalDataRatio_;
  DataConfig subConfig;
  int subDataProviderCount = config.sub_data_configs_size();
  if (isTestMode()) {
    LOG(INFO) << "construct MultiDataProvider in test mode";
  } else {
    LOG(INFO) << "construct MultiDataProvider in train mode";
  }
  subDataProviders_.resize(subDataProviderCount);
  for (int i = 0; i < subDataProviderCount; i++) {
    subConfig = config.sub_data_configs(i);
    if (subConfig.async_load_data()) {
      LOG(INFO) << "can not use async_load_data in sub dataprovider of "
                   "MultiDataProvider";
      subConfig.set_async_load_data(false);
    }
    subDataProviders_[i] = std::unique_ptr<DataProvider>(
        DataProvider::create(subConfig, modelConfig, useGpu_));
  }
}

void MultiDataProvider::reset() {
  for (auto& elem : subDataProviders_) {
    elem->reset();
  }
  DataProvider::reset();
}

void MultiDataProvider::shuffle() {
  for (auto& elem : subDataProviders_) {
    elem->shuffle();
  }
}

int64_t MultiDataProvider::getNextBatchInternal(int64_t size,
                                                DataBatch* batch) {
  batch->clear();
  for (size_t i = 0; i < subDataProviders_.size(); ++i) {
    // calc size according to data ratio
    int64_t subSize =
        (int64_t)(1.0 * size * config_.sub_data_configs(i).data_ratio() /
                  totalDataRatio_);
    DataBatch subBatch;
    int64_t realSize =
        subDataProviders_[i]->getNextBatchInternal(subSize, &subBatch);
    if (realSize == 0) {
      // current subDataProvider has no data
      if (!isTestMode()) {
        // in train mode
        if (config_.sub_data_configs(i).is_main_data()) {
          // is main data provider. then return 0
          batch->clear();
          return 0;
        } else {
          // not main data provider, reset current subDataProvider and try again
          subDataProviders_[i]->reset();
          subBatch.clear();
          realSize =
              subDataProviders_[i]->getNextBatchInternal(subSize, &subBatch);
          CHECK_GT(realSize, 0);
        }
      } else {
        // in test mode, make an empty argument
        Argument emptyArgu;
        std::vector<Argument> argus;
        argus.push_back(emptyArgu);
        batch->appendArguments(argus, 0, -1);
        continue;
      }
    }
    batch->appendArguments(subBatch.getStreams(), subBatch.getSize(), i);
  }
  return batch->getSize();
}

REGISTER_DATA_PROVIDER_EX(multi, MultiDataProvider);

}  // namespace paddle
