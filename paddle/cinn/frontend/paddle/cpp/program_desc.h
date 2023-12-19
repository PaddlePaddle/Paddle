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

#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "paddle/cinn/frontend/paddle/cpp/block_desc.h"
#include "paddle/cinn/frontend/paddle/cpp/desc_api.h"

namespace cinn::frontend::paddle::cpp {

/*
 * The cpp::ProgramDesc is the internal representation for Op. All the internal
 * imprementation should use it, not the pb::ProgramDesc.
 */
class ProgramDesc : public ProgramDescAPI {
 public:
  ProgramDesc() = default;

  size_t BlocksSize() const override { return blocks_.size(); }

  void ClearBlocks() override { blocks_.clear(); }

  template <typename T>
  T* GetBlock(int32_t idx);

  template <typename T>
  const T& GetConstBlock(int32_t idx) const;

  template <typename T>
  T* AddBlock();

  // Just return default versoin
  // TODO(sangoly): refine this
  bool HasVersion() const override { return true; }

  int64_t Version() const override { return version_; }

  void SetVersion(int64_t version) override { version_ = version; }

 private:
  int64_t version_;
  std::vector<cpp::BlockDesc> blocks_;
};

}  // namespace cinn::frontend::paddle::cpp
