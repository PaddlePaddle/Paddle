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

#include "ParameterUpdaterBase.h"
#include <fstream>
#include "hl_gpu.h"
#include "paddle/utils/Logging.h"

namespace paddle {

void ParameterUpdater::init(const std::vector<ParameterPtr>& parameters) {
  parameters_ = parameters;
  for (ParameterType type : getParameterTypes()) {
    for (auto& para : parameters) {
      para->enableType(type);
    }
  }
  for (size_t pid = 0; pid < parameters_.size(); ++pid) {
    nonStaticParaIDMap_.insert(
        std::pair<size_t, size_t>(parameters_[pid]->getID(), pid));
  }

  for (auto& para : parameters) {
    if (!para->isStatic()) {
      para->initHook();
    }
  }
}

}  // namespace paddle
