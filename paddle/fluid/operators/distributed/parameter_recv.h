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

#include <string>
#include <vector>

#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {
namespace distributed {

template <typename T>
struct ParameterRecv {
  void operator()(const std::string &var_name,
                  const std::vector<std::string> &send_varnames,
                  const std::vector<std::string> &epmap,
                  const std::vector<int64_t> &height_sections,
                  const framework::ExecutionContext &context,
                  const framework::Scope &scope, bool sync);
};

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
