/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/data_feed_factory.h"

#include <cstdlib>

#include <memory>
#include <string>

#include "glog/logging.h"

namespace paddle::framework {
class DataFeed;

typedef std::shared_ptr<DataFeed> (*Createdata_feedFunction)();
typedef std::unordered_map<std::string, Createdata_feedFunction> data_feedMap;
data_feedMap g_data_feed_map;

#define REGISTER_DATAFEED_CLASS(data_feed_class)                      \
  namespace {                                                         \
  std::shared_ptr<DataFeed> Creator_##data_feed_class() {             \
    return std::shared_ptr<DataFeed>(new data_feed_class);            \
  }                                                                   \
  class __Registerer_##data_feed_class {                              \
   public:                                                            \
    __Registerer_##data_feed_class() {                                \
      g_data_feed_map[#data_feed_class] = &Creator_##data_feed_class; \
    }                                                                 \
  };                                                                  \
  __Registerer_##data_feed_class g_registerer_##data_feed_class;      \
  }  // namespace

std::string DataFeedFactory::DataFeedTypeList() {
  std::string data_feed_types;
  for (auto iter = g_data_feed_map.begin(); iter != g_data_feed_map.end();
       ++iter) {
    if (iter != g_data_feed_map.begin()) {
      data_feed_types += ", ";
    }
    data_feed_types += iter->first;
  }
  return data_feed_types;
}

std::shared_ptr<DataFeed> DataFeedFactory::CreateDataFeed(
    std::string data_feed_class) {
  if (g_data_feed_map.count(data_feed_class) < 1) {
    LOG(WARNING) << "Your DataFeed " << data_feed_class
                 << " is not supported currently";
    LOG(WARNING) << " Supported DataFeed: " << DataFeedTypeList();
    exit(-1);
  }
  return g_data_feed_map[data_feed_class]();
}

REGISTER_DATAFEED_CLASS(MultiSlotDataFeed);
REGISTER_DATAFEED_CLASS(MultiSlotInMemoryDataFeed);
REGISTER_DATAFEED_CLASS(PaddleBoxDataFeed);
REGISTER_DATAFEED_CLASS(SlotRecordInMemoryDataFeed);
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && !defined(_WIN32)
REGISTER_DATAFEED_CLASS(MultiSlotFileInstantDataFeed);
#endif
}  // namespace paddle::framework
