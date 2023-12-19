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

#include <vector>

#include "paddle/cinn/frontend/paddle/cpp/desc_api.h"
#include "paddle/cinn/frontend/paddle/cpp/op_desc.h"
#include "paddle/cinn/frontend/paddle/cpp/var_desc.h"

namespace cinn::frontend::paddle::cpp {

/*
 * The cpp::BlockDesc is the internal representation for Op. All the internal
 * imprementation should use it, not the pb::BlockDesc.
 */
class BlockDesc : public BlockDescAPI {
 public:
  BlockDesc() = default;

  int32_t Idx() const override { return idx_; }

  void SetIdx(int32_t idx) override { idx_ = idx; }

  int32_t ParentIdx() const override { return parent_idx_; }

  void SetParentIdx(int32_t idx) override { parent_idx_ = idx; }

  size_t VarsSize() const override { return vars_.size(); }

  void ClearVars() override { vars_.clear(); }

  template <typename T>
  T* GetVar(int32_t idx);

  template <typename T>
  const T& GetConstVar(int32_t idx) const;

  template <typename T>
  T* AddVar();

  size_t OpsSize() const override { return ops_.size(); }

  void ClearOps() override { ops_.clear(); }

  template <typename T>
  T* GetOp(int32_t idx);

  template <typename T>
  const T& GetConstOp(int32_t idx) const;

  template <typename T>
  T* AddOp();

  int32_t ForwardBlockIdx() const override { return forward_block_idx_; }

  void SetForwardBlockIdx(int32_t idx) override { forward_block_idx_ = idx; }

 private:
  int32_t idx_;
  int32_t parent_idx_;
  std::vector<OpDesc> ops_;
  std::vector<VarDesc> vars_;
  int32_t forward_block_idx_;
};

}  // namespace cinn::frontend::paddle::cpp
