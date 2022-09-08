// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/strides.h"
#include <set>

namespace phi {

std::set<std::string> op_support_strides_list = {"transpose"};

bool Strides::IsOpSupportStrides(const std::string& op_type) {
  return op_support_strides_list.count(op_type) != 0;
}

void Strides::InitStrides(const DDim& ddim) {
  rank_ = ddim.size();
  strides_[ddim.size() - 1] = 1;
  for (int i = ddim.size() - 2; i >= 0; --i) {
    strides_[i] = strides_[i + 1] * ddim[i + 1];
  }
  is_valiable_ = true;
}

}  // namespace phi
