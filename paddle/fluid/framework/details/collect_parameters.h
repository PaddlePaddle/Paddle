//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/framework/channel.h"

namespace paddle {
namespace framework {
namespace details {

class ParameterCollection {
 public:
  explicit ParameterCollection(const std::vector<std::string> &para_names,
                               const int device_count);

  static ParameterCollection &Instance() {
    PADDLE_ENFORCE_NOT_NULL(param_collect,
                            "Need to Create ParameterCollection first!");
    return *param_collect;
  }

  /*! \brief  Create should only called by Init function */
  static ParameterCollection &Init(const std::vector<std::string> &para_names,
                                   const int device_count) {
    if (param_collect == nullptr) {
      param_collect = new ParameterCollection(para_names, device_count);
    }
    return *param_collect;
  }

  ChannelHolder *Get(const std::string &para_name);

  size_t size() const { return param_channels_.size(); }

 private:
  static ParameterCollection *param_collect;
  const int device_count_;
  std::unordered_map<const std::string, std::unique_ptr<ChannelHolder>>
      param_channels_;
  DISABLE_COPY_AND_ASSIGN(ParameterCollection);
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
