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

#include "paddle/framework/operator.h"

namespace paddle {
namespace framework {

std::string OperatorBase::DebugString() const {
  std::stringstream ss;
  ss << "Op(" << Type() << "), inputs:(";
  for (size_t i = 0; i < inputs_.size(); ++i) {
    ss << inputs_[i];
    if (i != inputs_.size() - 1) {
      ss << ", ";
    }
  }
  ss << "), outputs:(";
  for (size_t i = 0; i < outputs_.size(); ++i) {
    ss << outputs_[i];
    if (i != outputs_.size() - 1) {
      ss << ", ";
    }
  }
  ss << ").";
  return ss.str();
}

}  // namespace framework
}  // namespace paddle
