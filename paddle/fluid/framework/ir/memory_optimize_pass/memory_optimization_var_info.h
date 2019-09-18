// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

class MemOptVarInfo {
 public:
  MemOptVarInfo(const std::string &name, size_t ref_cnt) : name_(name) {
    SetRefCnt(ref_cnt);
  }

  bool DecreaseRefCnt() {
    return ref_cnt_ == 1 || (runtime_ref_cnt_.fetch_sub(1) == 1);
  }

  void ResetRuntimeRefCnt() { runtime_ref_cnt_ = ref_cnt_; }

  void SetRefCnt(size_t ref_cnt) {
    PADDLE_ENFORCE_GE(ref_cnt, 1,
                      "Reference count must be larger than or equal to 1");
    ref_cnt_ = ref_cnt;
    runtime_ref_cnt_ = ref_cnt;
  }

  bool IsSkipped() const { return skipped_; }

  void SetSkip(bool skipped) { skipped_ = skipped; }

  const std::string &Name() const { return name_; }

 private:
  std::string name_;
  size_t ref_cnt_;
  std::atomic<size_t> runtime_ref_cnt_;
  bool skipped_{false};
};

using MemOptVarInfoMapList = std::vector<
    std::unordered_map<std::string, std::unique_ptr<MemOptVarInfo>>>;

class SkipMemOptVarsGuard {
 public:
  SkipMemOptVarsGuard(MemOptVarInfoMapList *list,
                      const std::vector<std::string> &vars,
                      bool need_reset_ref_cnt)
      : list_(list), need_reset_ref_cnt_(need_reset_ref_cnt) {
    if (!list_) return;

    skip_vars_.reserve(vars.size() * list->size());
    for (auto &var : vars) {
      for (auto &map : *list_) {
        auto iter = map.find(var);
        if (iter != map.end() && !iter->second->IsSkipped()) {
          iter->second->SetSkip(true);
          skip_vars_.emplace_back(iter->second.get());
        }
      }
    }
  }

  ~SkipMemOptVarsGuard() {
    for (auto *var : skip_vars_) {
      var->SetSkip(false);
    }

    if (list_ && need_reset_ref_cnt_) {
      for (auto &map : *list_) {
        for (auto &pair : map) {
          pair.second->ResetRuntimeRefCnt();
        }
      }
    }
  }

 private:
  MemOptVarInfoMapList *list_;
  bool need_reset_ref_cnt_;
  std::vector<MemOptVarInfo *> skip_vars_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
