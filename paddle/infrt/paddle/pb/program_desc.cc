// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/paddle/pb/program_desc.h"

#include <algorithm>
#include <limits>

namespace infrt::paddle::pb {

template <>
framework_proto::BlockDesc* ProgramDesc::GetBlock<framework_proto::BlockDesc>(
    int32_t idx) {
  CHECK_LT(idx, static_cast<int>(BlocksSize())) << "idx >= blocks.size()";
  return desc_->mutable_blocks(idx);
}

template <>
framework_proto::BlockDesc*
ProgramDesc::AddBlock<framework_proto::BlockDesc>() {
  return desc_->add_blocks();
}

}  // namespace infrt::paddle::pb
