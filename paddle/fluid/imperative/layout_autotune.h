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

#pragma once
#include <glog/logging.h>
#include <memory>
#include <unordered_set>
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/compat/type_defs.h"

namespace paddle {
namespace imperative {

class Tracer;

using DataLayout = paddle::experimental::DataLayout;

class LayoutAutoTune {
 public:
  static LayoutAutoTune& Instance() {
    static LayoutAutoTune layout_autoTune;
    return layout_autoTune;
  }

  bool UseLayoutAutoTune() const;

  void EnableLayoutAutoTune() { use_layout_autotune_ = true; }

  void DisableLayoutAutoTune() { use_layout_autotune_ = false; }

  bool IsLightlyLayoutSensitive(const std::string& op_type) const {
    return lightly_layout_sensitive_ops_.count(op_type) != 0;
  }

  bool IsLayoutAgnostic(const std::string& op_type) const {
    return layout_agnostic_ops_.count(op_type) != 0;
  }

  DataLayout GetDesiredLayout() const { return layout_; }

  void SetDesiredLayout(const DataLayout& layout) { layout_ = layout; }

 private:
  LayoutAutoTune();

  bool use_layout_autotune_{false};

  std::unordered_set<std::string> layout_agnostic_ops_{};

  std::unordered_set<std::string> heavily_layout_sensitive_ops_{};

  std::unordered_set<std::string> lightly_layout_sensitive_ops_{};

  DataLayout layout_{DataLayout::UNDEFINED};
};

template <typename VarType>
paddle::imperative::NameVarMap<VarType> AutoTuneLayout(
    const std::string& op_type,
    const paddle::imperative::NameVarMap<VarType>& ins,
    const paddle::imperative::NameVarMap<VarType>& outs,
    paddle::framework::AttributeMap* attrs,
    const std::shared_ptr<imperative::Tracer>& tracer);

}  // namespace imperative
}  // namespace paddle
