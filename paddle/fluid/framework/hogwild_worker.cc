/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/lodtensor_printer.h"

namespace paddle {
namespace framework {

void HogwildWorker::Initialize(const TrainerDesc& desc) {
  fetch_config_ = desc.fetch_config();
  param_ = desc.hogwild_param();
  skip_ops_.resize(param_.skip_ops_size());
  LOG(WARNING) << "skip op size: " << skip_ops_.size();
  for (size_t i = 0; i < param_.skip_ops_size(); ++i) {
    skip_ops_[i] = param_.skip_ops(i);
  }
}

void HogwildWorker::CreateThreadOperators(const ProgramDesc& program) {
  auto& block = program.Block(0);
  op_names_.clear();
  for (auto& op_desc : block.AllOps()) {
    std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
    op_names_.push_back(op_desc->Type());
    OperatorBase* local_op_ptr = local_op.release();
    ops_.push_back(local_op_ptr);
    continue;
  }
}

void HogwildWorker::CreateThreadScope(const ProgramDesc& program) {
  auto& block = program.Block(0);

  PADDLE_ENFORCE_NOT_NULL(
      root_scope_, "root_scope should be set before creating thread scope");

  thread_scope_ = &root_scope_->NewScope();
  for (auto& var : block.AllVars()) {
    if (var->Persistable()) {
      auto* ptr = root_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
    } else {
      auto* ptr = thread_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
    }
  }
}

void HogwildWorker::BindingDataFeedMemory() {
  const std::vector<std::string>& input_feed =
      device_reader_->GetUseSlotAlias();
  for (auto name : input_feed) {
    device_reader_->AddFeedVar(thread_scope_->Var(name), name);
  }
}

void HogwildWorker::CreateDeviceResource(const ProgramDesc& main_prog) {
  CreateThreadScope(main_prog);
  CreateThreadOperators(main_prog);
}

void HogwildWorker::TrainFilesWithProfiler() {
  platform::SetNumThreads(1);
  device_reader_->Start();
  std::vector<double> op_total_time;
  std::vector<std::string> op_name;
  for (auto& op : ops_) {
    op_name.push_back(op->Type());
  }
  op_total_time.resize(ops_.size());
  for (size_t i = 0; i < op_total_time.size(); ++i) {
    op_total_time[i] = 0.0;
  }
  platform::Timer timeline;
  double total_time = 0.0;
  double read_time = 0.0;
  int cur_batch;
  int batch_cnt = 0;
  timeline.Start();
  uint64_t total_inst = 0;
  while ((cur_batch = device_reader_->Next()) > 0) {
    VLOG(3) << "read a batch in thread " << thread_id_;
    timeline.Pause();
    read_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();
    for (size_t i = 0; i < ops_.size(); ++i) {
      bool need_skip = false;
      for (auto t = 0u; t < skip_ops_.size(); ++t) {
        if (ops_[i]->Type().find(skip_ops_[t]) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      timeline.Start();
      VLOG(3) << "Going to run op " << op_name[i];
      if (!need_skip) {
        ops_[i]->Run(*thread_scope_, place_);
      }
      VLOG(3) << "Op " << op_name[i] << " Finished";
      timeline.Pause();
      op_total_time[i] += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
    }
    total_inst += cur_batch;
    ++batch_cnt;
    PrintFetchVars();
    if (thread_id_ == 0) {
      if (batch_cnt > 0 && batch_cnt % 100 == 0) {
        for (size_t i = 0; i < ops_.size(); ++i) {
          fprintf(stderr, "op_name:[%zu][%s], op_mean_time:[%fs]\n", i,
                  op_name[i].c_str(), op_total_time[i] / batch_cnt);
        }
        fprintf(stderr, "mean read time: %fs\n", read_time / batch_cnt);
        fprintf(stderr, "IO percent: %f\n", read_time / total_time * 100);
        fprintf(stderr, "%6.2f instances/s\n", total_inst / total_time);
      }
    }
    thread_scope_->DropKids();
    timeline.Start();
  }
}

void HogwildWorker::TrainFiles() {
  platform::SetNumThreads(1);

  // how to accumulate fetched values here
  device_reader_->Start();
  int cur_batch;
  while ((cur_batch = device_reader_->Next()) > 0) {
    for (auto& op : ops_) {
      bool need_skip = false;
      for (auto t = 0u; t < skip_ops_.size(); ++t) {
        if (op->Type().find(skip_ops_[t]) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      if (!need_skip) {
        op->Run(*thread_scope_, place_);
      }
    }

    PrintFetchVars();
    thread_scope_->DropKids();
  }
}

void HogwildWorker::PrintFetchVars() {
  // call count
  batch_num_++;
  int batch_per_print = fetch_config_.print_period();
  if (thread_id_ == 0) {
    if (batch_num_ % batch_per_print == 0) {
      int fetch_var_num = fetch_config_.fetch_var_names_size();
      for (int i = 0; i < fetch_var_num; ++i) {
        platform::PrintVar(thread_scope_, fetch_config_.fetch_var_names(i),
                           fetch_config_.fetch_var_str_format(i));
      }
    }
  }
}

}  // end namespace framework
}  // end namespace paddle
