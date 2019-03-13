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

#include "paddle/fluid/framework/dataset_factory.h"
#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/data_set.h"

namespace paddle {
namespace framework {
typedef std::shared_ptr<Dataset> (*CreateDatasetFunction)();
typedef std::unordered_map<std::string, CreateDatasetFunction> datasetMap;
datasetMap g_dataset_map;

#define REGISTER_DATASET_CLASS(dataset_class)                      \
  namespace {                                                         \
  std::shared_ptr<Dataset> Creator_##dataset_class() {             \
    return std::shared_ptr<Dataset>(new dataset_class);            \
  }                                                                   \
  class __Registerer_##dataset_class {                              \
   public:                                                            \
    __Registerer_##dataset_class() {                                \
      g_dataset_map[#dataset_class] = &Creator_##dataset_class; \
    }                                                                 \
  };                                                                  \
  __Registerer_##dataset_class g_registerer_##dataset_class;      \
  }  // namespace

std::string DatasetFactory::DatasetTypeList() {
  std::string dataset_types;
  for (auto iter = g_dataset_map.begin(); iter != g_dataset_map.end();
       ++iter) {
    if (iter != g_dataset_map.begin()) {
      dataset_types += ", ";
    }
    dataset_types += iter->first;
  }
  return dataset_types;
}

std::shared_ptr<Dataset> DatasetFactory::CreateDataset(
    std::string dataset_class) {
  if (g_dataset_map.count(dataset_class) < 1) {
    LOG(WARNING) << "Your Dataset " << dataset_class
                 << "is not supported currently";
    LOG(WARNING) << "Supported Dataset: " << DatasetTypeList();
    exit(-1);
  }
  return g_dataset_map[dataset_class]();
}

REGISTER_DATASET_CLASS(MultiSlotDataset);
}  // namespace framework
}  // namespace paddle
