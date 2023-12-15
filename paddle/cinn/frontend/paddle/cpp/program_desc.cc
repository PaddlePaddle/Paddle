// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/frontend/paddle/cpp/program_desc.h"

namespace cinn::frontend::paddle::cpp {

template <>
BlockDesc* ProgramDesc::GetBlock<BlockDesc>(int32_t idx) {
  CHECK_LT(idx, BlocksSize()) << "idx >= blocks.size()";
  return &blocks_[idx];
}

template <>
const BlockDesc& ProgramDesc::GetConstBlock<BlockDesc>(int32_t idx) const {
  CHECK_LT(idx, static_cast<int32_t>(BlocksSize())) << "idx >= blocks.size()";
  return blocks_[idx];
}

template <>
BlockDesc* ProgramDesc::AddBlock<BlockDesc>() {
  blocks_.emplace_back();
  return &blocks_.back();
}

}  // namespace cinn::frontend::paddle::cpp
