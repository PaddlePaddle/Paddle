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

#pragma once

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>
#include "paddle/framework/op_desc.h"
#include "paddle/framework/var_desc.h"
#include "paddle/platform/macros.h"

namespace paddle {
namespace framework {

class ProgramDescBind;

// Each Protobuf Message, we provide a XXXBind class. In that class, we optimize
// read/write speed. Only when we want the protobuf message, the local changes
// will be synchronized (by `Sync` method).

class BlockDescBind {
 public:
  BlockDescBind(ProgramDescBind *prog, BlockDesc *desc)
      : prog_(prog), desc_(desc), need_update_(false) {}

  int32_t ID() const { return desc_->idx(); }

  int32_t Parent() const { return desc_->parent_idx(); }

  VarDescBind *Var(const std::string &name_bytes);

  VarDescBind *FindVar(const std::string &name_bytes) const;

  bool HasVar(const std::string &var_name) const;

  std::vector<VarDescBind *> AllVars() const;

  BlockDescBind *ParentBlock() const;

  OpDescBind *AppendOp();

  OpDescBind *PrependOp();

  std::vector<OpDescBind *> AllOps() const;

  void Flush();

  BlockDesc *Proto();

  // FIXME(yuyang18): backward will access private data of BlockDesc.
  // Mark it public temporary. We can fix it later.
 public:
  ProgramDescBind *prog_;  // not_own
  BlockDesc *desc_;        // not_own
  bool need_update_;

  std::deque<std::unique_ptr<OpDescBind>> ops_;
  std::unordered_map<std::string, std::unique_ptr<VarDescBind>> vars_;

  DISABLE_COPY_AND_ASSIGN(BlockDescBind);
};
}  // namespace framework
}  // namespace paddle
