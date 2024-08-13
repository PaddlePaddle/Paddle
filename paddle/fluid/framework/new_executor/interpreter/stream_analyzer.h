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
#include <future>
#include <memory>
#include <vector>

#include "paddle/fluid/framework/new_executor/interpreter/dependency_builder.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/platform/device_event.h"

namespace paddle {
namespace framework {
namespace interpreter {

enum DownstreamRunType { kDirectRun, kEventRun };

class ContextManager {
 public:
  using DeviceContextMap =
      std::map<Place, std::shared_future<std::unique_ptr<phi::DeviceContext>>>;

  static ContextManager& Instance() {
    static ContextManager* ctx_manager = new ContextManager;
    return *ctx_manager;
  }

  std::shared_future<std::unique_ptr<phi::DeviceContext>> Get(
      const std::string& type, const phi::Place& place, int stream_priority) {
    std::lock_guard<std::mutex> lk(ctx_mtx_);
    VLOG(6) << "Get dev_ctx for " << type << " - " << place;

    DeviceContextMap& ctxs = ctx_pool_[type];
    if (ctxs.find(place) == ctxs.end()) {
      platform::EmplaceDeviceContexts(
          &ctxs,
          {place},
          /*disable_setting_default_stream_for_allocator=*/true,
          stream_priority);
    }
    return ctxs[place];
  }

 private:
  ContextManager() {}
  DISABLE_COPY_AND_ASSIGN(ContextManager);

  std::mutex ctx_mtx_;
  std::unordered_map<std::string, DeviceContextMap> ctx_pool_;
};

class StreamAnalyzer {
 public:
  using DeviceContext = phi::DeviceContext;
  using Place = phi::Place;

  explicit StreamAnalyzer(const Place& place) : place_(place) {
    event_info_ = std::make_shared<
        std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>>();
  }

  ~StreamAnalyzer() {}

  void ConstructEvents(std::vector<Instruction>* instructions);

  phi::DeviceContext* ParseDeviceContext(const OpFuncNode& op_func_node) const;

  platform::DeviceType GetWaiterType(const Instruction& instr) const;

  void ShareEventInfoFrom(const StreamAnalyzer& src);

  void SetForceEventsToWaitInfo(
      std::unordered_map<std::string, std::shared_ptr<EventInter>>*
          program_force_events_to_wait) {
    program_force_events_to_wait_ = program_force_events_to_wait;
  }

  std::shared_ptr<
      std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>>
  GetEventInfo() const;

 private:
  bool HasDataDependency(Instruction* cur_instr, Instruction* next_instr) const;

  void AnalyseAllEventInfo(
      const std::vector<Instruction*>& instructions,
      const std::vector<std::vector<std::vector<size_t>>>& run_type_info,
      std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>*
          event_info) const;

  void AnalyseAllRunType(
      const std::vector<Instruction*>& instructions,
      const std::map<size_t, std::set<size_t>>& downstream_map,
      std::vector<std::vector<std::vector<size_t>>>* run_type_info) const;

  void ShrinkEventInfo(
      const DependencyBuilder& dependency_builder,
      std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>*
          event_info_map) const;

  const Place place_;
  bool is_event_info_build_{false};
  std::shared_ptr<
      std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>>
      event_info_;
  std::unordered_map<std::string, std::shared_ptr<EventInter>>*
      program_force_events_to_wait_;  // not owned
};

/// ======================== ///
///        For new ir        ///
/// ======================== ///
class PirStreamAnalyzer {
 public:
  using DeviceContext = phi::DeviceContext;
  using Place = phi::Place;

  explicit PirStreamAnalyzer(const Place& place) : place_(place) {
    event_info_ = std::make_shared<
        std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>>();
  }

  ~PirStreamAnalyzer() {}

  void ConstructEvents(
      const std::vector<std::unique_ptr<paddle::framework::InstructionBase>>&
          instructions);

  platform::DeviceType GetWaiterType(
      const paddle::framework::InstructionBase* instr) const;

  void ShareEventInfoFrom(const PirStreamAnalyzer& src);

  void SetForceEventsToWaitInfo(
      std::unordered_map<std::string, std::shared_ptr<EventInter>>*
          program_force_events_to_wait) {
    program_force_events_to_wait_ = program_force_events_to_wait;
  }

  std::shared_ptr<
      std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>>
  GetEventInfo() const;

 private:
  void AnalyseAllRunType(
      const std::vector<paddle::framework::InstructionBase*>& instructions,
      const std::map<size_t, std::set<size_t>>& downstream_map,
      std::vector<std::vector<std::vector<size_t>>>* run_type_info) const;

  void AnalyseAllEventInfo(
      const std::vector<paddle::framework::InstructionBase*>& instructions,
      const std::vector<std::vector<std::vector<size_t>>>& run_type_info,
      std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>*
          event_info) const;

  void ShrinkEventInfo(
      const PirDependencyBuilder& dependency_builder,
      std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>*
          event_info_map) const;

  const Place place_;
  bool is_event_info_build_{false};
  std::shared_ptr<
      std::map<const DeviceContext*, std::map<size_t, std::set<size_t>>>>
      event_info_;
  std::unordered_map<std::string, std::shared_ptr<EventInter>>*
      program_force_events_to_wait_;  // not owned
};

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
