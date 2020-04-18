/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"

namespace paddle {
namespace imperative {

class AutoCastGuard {
 public:
  explicit inline AutoCastGuard(std::shared_ptr<Tracer> tracer, bool guard_mode)
      : tracer_(tracer), guard_mode_(guard_mode) {
    pre_mode_ = tracer_->IsAutoCastEnabled();
    if (pre_mode_ != guard_mode) {
      tracer_->SetEnableAutoCast(guard_mode_);
    }
  }

  inline ~AutoCastGuard() { tracer_->SetEnableAutoCast(pre_mode_); }

  AutoCastGuard(const AutoCastGuard& guard) = delete;
  AutoCastGuard& operator=(const AutoCastGuard& guard) = delete;

 private:
  std::shared_ptr<Tracer> tracer_;
  bool guard_mode_;
  bool pre_mode_;
};

NameVarBaseMap AutoCastInputs(const std::string& op_type,
                              const NameVarBaseMap& ins);
}  // namespace imperative
}  // namespace paddle
