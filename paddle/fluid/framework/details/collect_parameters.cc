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

#include <unordered_set>

#include "paddle/fluid/framework/details/collect_parameters.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace framework {
namespace details {

ParameterCollection *ParameterCollection::param_collect = nullptr;

ParameterCollection::ParameterCollection(
    const std::vector<std::string> &para_names, const int device_count)
    : device_count_(device_count) {
  PADDLE_ENFORCE_GT(para_names.size(), 0);
  PADDLE_ENFORCE_NOT_NULL(param_collect,
                          "Need to Create ParameterCollection first!");
  using PtrType = std::unique_ptr<ChannelHolder>;
  for (size_t i = 0; i < para_names.size(); ++i) {
    param_channels_.emplace(para_names[i], PtrType(new ChannelHolder()));
    param_channels_[para_names[i]]->Reset<Variable *>(device_count);
  }
}

ChannelHolder *ParameterCollection::Get(const std::string &param_name) {
  PADDLE_ENFORCE_NOT_NULL(param_collect,
                          "Need to Create ParameterCollection first!");
  auto it = param_channels_.find(param_name);
  if (it == param_channels_.end()) {
    PADDLE_THROW("%s is in the ParameterCollection.", param_name);
  }
  return it->second.get();
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
