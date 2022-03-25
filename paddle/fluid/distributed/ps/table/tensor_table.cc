// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/ps/table/tensor_table.h"

DECLARE_double(eager_delete_tensor_gb);
namespace paddle {
namespace distributed {

int32_t TensorTable::set_program_env(
    framework::Scope *scope, platform::Place place,
    const std::vector<framework::ProgramDesc> *sub_program) {
  scope_ = scope;
  place_ = place;
  executor_ = new framework::Executor(place_);
  sub_program_ = sub_program;
  return 0;
}

int32_t GlobalStepTable::initialize() {
  auto _program_config = _config.tensor();
  auto trainers_ = _config.common().trainer_num();
  FLAGS_eager_delete_tensor_gb = -1;
  // Get Config
  if (_program_config.has_startup_program_id()) {
    startup_program_id_ = _program_config.startup_program_id();
  }
  if (_program_config.has_main_program_id()) {
    main_program_id_ = _program_config.main_program_id();
  }
  if (_program_config.has_feed_var_name()) {
    feed_var_name_ = _program_config.feed_var_name();
  }
  if (_program_config.has_fetch_var_name()) {
    fetch_var_name_ = _program_config.fetch_var_name();
  }

  // Run startup program
  if (startup_program_id_ != -1) {
    std::map<std::string, const framework::LoDTensor *> fake_feed;
    std::map<std::string, framework::FetchType *> fake_fetch;
    auto startup_program_desc = sub_program_->at(startup_program_id_);
    auto ctx = executor_->Prepare(startup_program_desc, 0);
    executor_->RunPreparedContext(ctx.get(), scope_, false);
  }

  if (main_program_id_ != -1) {
    // Run main porgram, if program is used for learning decay
    auto main_program_desc = sub_program_->at(main_program_id_);
    auto main_ctx = executor_->Prepare(main_program_desc, 0);
    exec_context_ = std::move(main_ctx);
    executor_->RunPreparedContext(exec_context_.get(), scope_, false);
    // init decay_counters
    decay_counters_.reserve(trainers_);
    for (int32_t i = 0; i < trainers_; ++i) {
      decay_counters_[i] = 0;
    }
  }

  return 0;
}

int32_t GlobalStepTable::set_table_map(
    std::unordered_map<uint32_t, std::shared_ptr<Table>> *table_map) {
  auto *lr_var = scope_->FindVar(fetch_var_name_);
  auto *lr_tensor = lr_var->GetMutable<framework::LoDTensor>();
  auto *lr_value = lr_tensor->mutable_data<float>(platform::CPUPlace());
  VLOG(3) << "GlobalStepTable::set_table_map set global lr: " << *lr_value;

  for (auto iter = table_map->begin(); iter != table_map->end(); iter++) {
    auto table_id = iter->first;
    if (table_id == _config.table_id()) {
      continue;
    }
    iter->second->set_global_lr(lr_value);
  }
  return 0;
}

int32_t GlobalStepTable::push_dense(const int64_t *values,
                                    const int32_t trainer_id) {
  return _run_program(values, trainer_id);
}

int32_t GlobalStepTable::_run_program(const int64_t *values,
                                      const uint32_t trainer_id) {
  FLAGS_eager_delete_tensor_gb = -1;
  auto counter = decay_counters_.at(trainer_id);
  counter += int(values[0]);
  decay_counters_.at(trainer_id) = counter;

  auto *global_step_var = scope_->FindVar(feed_var_name_);
  auto *tensor = global_step_var->GetMutable<framework::LoDTensor>();
  auto *value = tensor->mutable_data<int64_t>(platform::CPUPlace());

  auto global_counter = 0;
  for (auto &trainer_counter : decay_counters_) {
    global_counter += trainer_counter.second;
  }

  // Todo: hard code for increment op
  value[0] = global_counter - 1;
  VLOG(3) << "GlobalStepTable::_run_program global_counter " << value[0];

  executor_->RunPreparedContext(exec_context_.get(), scope_, false, false);
  auto *lr_var = scope_->FindVar(fetch_var_name_);
  auto *lr_tensor = lr_var->GetMutable<framework::LoDTensor>();
  auto *lr_value = lr_tensor->mutable_data<float>(platform::CPUPlace());
  VLOG(3) << "GlobalStepTable::LR value: " << lr_value[0];
  return 0;
}

}  // namespace distributed
}  // namespace paddle
