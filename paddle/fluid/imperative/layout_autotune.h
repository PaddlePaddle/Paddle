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

#include "paddle/common/layout.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/imperative/tracer.h"
namespace paddle {
namespace imperative {

class Tracer;

using DataLayout = phi::DataLayout;

class LayoutAutoTune {
 public:
  static LayoutAutoTune& Instance() {
    static LayoutAutoTune layout_autoTune;
    return layout_autoTune;
  }

  bool IsHeavilyLayoutSensitive(const std::string& op_type) const {
    return heavily_layout_sensitive_ops_.count(op_type) != 0;
  }

  bool IsLightlyLayoutSensitive(const std::string& op_type) const {
    return lightly_layout_sensitive_ops_.count(op_type) != 0;
  }

  bool IsLayoutAgnostic(const std::string& op_type) const {
    return layout_agnostic_ops_.count(op_type) != 0;
  }

  DataLayout GetDesiredLayout() const { return desired_layout_; }

  DataLayout GetDefaultLayout() const { return default_layout_; }

  void SetDesiredLayout(const DataLayout& layout) { desired_layout_ = layout; }

  void SetDefaultLayout(const DataLayout& layout) { default_layout_ = layout; }

 private:
  LayoutAutoTune();

  std::unordered_set<std::string> layout_agnostic_ops_{};

  std::unordered_set<std::string> heavily_layout_sensitive_ops_{"batch_norm"};

  std::unordered_set<std::string> lightly_layout_sensitive_ops_{
      "instance_norm", "softmax", "transpose", "transpose2", "reshape2"};

  // Best Layout in this platform
  DataLayout desired_layout_{DataLayout::UNDEFINED};

  // Default Layout in this model
  DataLayout default_layout_{DataLayout::UNDEFINED};
};

// LayoutAutotuneGuard is used for RAII.
class LayoutAutotuneGuard {
 public:
  LayoutAutotuneGuard(std::shared_ptr<Tracer> tracer, bool use_autotune);

  ~LayoutAutotuneGuard();

  // forbid copy and operator=
  LayoutAutotuneGuard(const LayoutAutotuneGuard& guard) = delete;
  LayoutAutotuneGuard& operator=(const LayoutAutotuneGuard& guard) = delete;

 private:
  std::shared_ptr<Tracer> tracer_;
  bool pre_layout_autotune_;
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
