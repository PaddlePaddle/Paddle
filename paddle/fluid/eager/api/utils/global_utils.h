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
//

#pragma once

#include <atomic>
#include <memory>

#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/hooks.h"
#include "paddle/fluid/eager/type_defs.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/utils/small_vector.h"
#include "paddle/utils/test_macros.h"
namespace egr {
class UniqueNameGenerator {
 public:
  explicit UniqueNameGenerator(std::string prefix = "") : prefix_(prefix) {}
  std::string Generate(std::string key = "eager_tmp") {
    return prefix_ + key + "_" + std::to_string(id_++);
  }

 private:
  std::atomic<int> id_{0};
  std::string prefix_;
};

// Global
// TODO(jiabin): Now we are using imperative tracer, move it here when we
// deprecate imperative.

class GradNodeBase;

class Controller {
 public:
  TEST_API static Controller& Instance();

  phi::Place GetExpectedPlace() const { return tracer_->ExpectedPlace(); }
  TEST_API void SetExpectedPlace(const phi::Place& place);
  void SetAMPLevel(paddle::imperative::AmpLevel level) {
    tracer_->SetAmpLevel(level);
  }
  paddle::imperative::AmpLevel GetAMPLevel() const {
    return tracer_->GetAmpLevel();
  }

  TEST_API void SetUsePromote(bool use_promote);
  TEST_API bool GetUsePromote() const;

  TEST_API bool UseLayoutAutoTune();

  void DisableLayoutAutoTune() { tracer_->DisableLayoutAutoTune(); }

  void EnableLayoutAutoTune() { tracer_->EnableLayoutAutoTune(); }

  void SetPythonStack(std::string stack_str) {
    tracer_->SetPythonStack(stack_str);
  }

  std::string GetPythonStack() { return tracer_->GetPythonStack(); }

  bool HasGrad() const { return tracer_->HasGrad(); }

  void SetHasGrad(bool has_grad) { tracer_->SetHasGrad(has_grad); }

  std::string GenerateUniqueName(std::string key = "eager_in_tmp") {
    return tracer_->GenerateUniqueName(key);
  }

  const std::shared_ptr<paddle::imperative::Tracer>& GetCurrentTracer() {
    return tracer_;
  }

  void SetCurrentTracer(
      const std::shared_ptr<paddle::imperative::Tracer>& tracer) {
    tracer_ = tracer;
    VLOG(6) << "Set current tracer for Controller: " << tracer_;
  }

  const std::shared_ptr<paddle::imperative::AmpAttrs>& GetCurrentAmpAttrs() {
    return paddle::imperative::GetCurrentAmpAttrs();
  }

  const std::unordered_map<std::string, std::vector<paddle::OpMetaInfo>>&
  GetOpMetaInfoMap() {
    return op_meta_info_map_;
  }

  void MergeOpMetaInfoMap(
      const std::unordered_map<std::string, std::vector<paddle::OpMetaInfo>>&
          map) {
    for (const auto& [key, value] : map) {
      if (op_meta_info_map_.count(key)) {
        VLOG(3) << "Replacing existing OpMetaInfo for op: " << key;
      }
      VLOG(3) << "Merging OpMetaInfo for op: " << key;
      op_meta_info_map_[key] = value;
    }
  }

  std::unordered_map<std::string,
                     std::vector<std::vector<std::unordered_map<int, int>>>>&
  GetCustomEdgesSlotMap() {
    return custom_edges_slot_map_;
  }
  // For Cpp Hook
  void RegisterBackwardFinalHook(const std::function<void()>& call_back) {
    VLOG(6) << "RegisterBackwardFinalHook";
    final_backward_hooks_.emplace_back(
        std::make_shared<CppVoidHook>(std::move(call_back)));
    VLOG(6) << "Size: " << final_backward_hooks_.size();
  }
  // For Python hook
  void RegisterBackwardFinalHook(const std::shared_ptr<VoidHook>& call_back) {
    final_backward_hooks_.emplace_back(call_back);
  }
  const std::vector<std::shared_ptr<VoidHook>>& FinalBackwardHooks() const {
    return final_backward_hooks_;
  }

  void ClearFinalBackwardHooks() { final_backward_hooks_.clear(); }

  void ClearForceSequentialNodes() { force_sequential_nodes_.clear(); }
  void PushBackForceSequentialNodes(GradNodeBase* node) {
    force_sequential_nodes_.push_back(node);
  }

  void EraseForceSequentialNodes(GradNodeBase* node) {
    for (auto iter = force_sequential_nodes_.begin();
         iter != force_sequential_nodes_.end();
         ++iter) {
      if (*iter == node) {
        force_sequential_nodes_.erase(iter);
        return;
      }
    }
  }

  std::list<GradNodeBase*> GetForceSequentialNodes() {
    return force_sequential_nodes_;
  }

  TEST_API void SetIsInBackward(bool is_in_backward);
  TEST_API bool GetIsInBackward() const;

 private:
  Controller() = default;
  static Controller* controller_;
  static thread_local std::shared_ptr<paddle::imperative::Tracer> tracer_;
  std::unordered_map<std::string, std::vector<paddle::OpMetaInfo>>
      op_meta_info_map_;
  /* op_type : {{{grad_outputs}, {grad_inputs}, {input}, {output}, {attrs}},
   * {{grad_outputs}, {grad_inputs}, {input}, {output}, {attrs}}}*/
  std::unordered_map<std::string,
                     std::vector<std::vector<std::unordered_map<int, int>>>>
      custom_edges_slot_map_;
  std::vector<std::shared_ptr<VoidHook>> final_backward_hooks_;
  std::list<GradNodeBase*> force_sequential_nodes_;
  bool is_in_backward_{false};
  DISABLE_COPY_AND_ASSIGN(Controller);
};

class EagerBackwardStateGuard {
 public:
  EagerBackwardStateGuard() { Controller::Instance().SetIsInBackward(true); }

  ~EagerBackwardStateGuard() { Controller::Instance().SetIsInBackward(false); }
};
class EagerBackwardSubGraphNodeRecorder {
 public:
  /**
   * @brief Get the singleton instance of EagerBackwardSubGraphNodeRecorder
   * @return Reference to the singleton instance
   *
   * Uses static local variable for thread-safe singleton initialization
   * (C++11 guarantee). The instance is created on first call and destroyed
   * automatically at program termination.
   */
  static EagerBackwardSubGraphNodeRecorder& Instance() {
    static EagerBackwardSubGraphNodeRecorder instance;
    return instance;
  }

 public:
  void AddGradNode(const GradNodeBase* node) { set_.insert(node); }
  void RemoveGradNode(const GradNodeBase* node) { set_.erase(node); }
  bool ContainsGradNode(const GradNodeBase* node) { return set_.count(node); }
  bool NeedCaptureSubGraph() { return need_capture_subgraph_; }
  void StartCaptureSubGraph() { need_capture_subgraph_ = true; }
  void EndCaptureSubGraph() { need_capture_subgraph_ = false; }
  void SetDumpDirPath(const std::string& path) { dump_dir_path_ = path; }
  const std::string& GetDumpDirPath() { return dump_dir_path_; }
  void SetNeedDumpGradTensors(bool need_dump) {
    need_dump_grad_tensors_ = need_dump;
  }
  bool GetNeedDumpGradTensors() { return need_dump_grad_tensors_; }
  bool HasCapturedSubgraph() { return !set_.empty(); }

 private:
  std::unordered_set<const GradNodeBase*> set_;
  std::string dump_dir_path_;
  bool need_dump_grad_tensors_ = false;
  bool need_capture_subgraph_ = false;
};

/**
 * @class LogIndent
 * @brief Singleton class for managing log indentation levels globally
 *
 * This class implements the singleton pattern to provide a centralized way
 * to manage indentation levels for formatted log output. It ensures only
 * one instance exists throughout the application lifecycle.
 */
class LogIndent {
 public:
  /**
   * @brief Get the singleton instance of LogIndent
   * @return Reference to the singleton instance
   *
   * Uses static local variable for thread-safe singleton initialization
   * (C++11 guarantee). The instance is created on first call and destroyed
   * automatically at program termination.
   */
  static LogIndent& Instance() {
    static LogIndent instance;
    return instance;
  }
  /**
   * @brief Increase the current indentation level by 1
   *
   * Call this method when entering a nested scope to increase
   * log indentation for better visual hierarchy.
   */
  void IncreaseIndentLevel() { FLAGS_indentlevel = FLAGS_indentlevel + 1; }
  /**
   * @brief Decrease the current indentation level by 1
   *
   * Reduces the indentation level, but never goes below 0.
   * Call this when leaving a nested scope.
   */
  void DecreaseIndentLevel() {
    if (FLAGS_indentlevel > 0) {
      FLAGS_indentlevel = FLAGS_indentlevel - 1;
    }
  }
  LogIndent(const LogIndent&) = delete;
  LogIndent& operator=(const LogIndent&) = delete;

 private:
  LogIndent() = default;
  ~LogIndent() = default;
};

}  // namespace egr
