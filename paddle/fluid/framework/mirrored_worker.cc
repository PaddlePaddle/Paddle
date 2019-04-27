/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

void MirroredWorker::Initialize(const TrainerDesc& desc) {
  fetch_config_ = desc.fetch_config();
  param_ = desc.mirrored_param();
  skip_ops_.resize(param_.skip_ops_size());
  for (int i = 0; i < param_.skip_ops_size(); ++i) {
    skip_ops_[i] = param_.skip_ops(i);
  }
  grad_names_.resize(param_.grad_names_size());
  for (int i = 0; i < param_.grad_names_size(); ++i) {
    grad_names_[i] = param_.grad_names(i);
  }
#ifdef PADDLE_WITH_CUDA
  nccl_ptr_ = NCCLWrapper::GetInstance();
  device_id_ = nccl_ptr_->nccl_info_.local_rank_;
  VLOG(3) << " nccl local rank: " << nccl_ptr_->nccl_info_.local_rank_
          << " global rank: " << nccl_ptr_->nccl_info_.my_global_rank_
          << " total ranks: " << nccl_ptr_->nccl_info_.global_ranks_;
#endif
}

void MirroredWorker::CreateThreadOperators(const ProgramDesc& program) {
  auto& block = program.Block(0);
  forward_backward_ops_.clear();
  optimize_ops_.clear();
  VLOG(3) << "begin to create operators";
  for (auto& op_desc : block.AllOps()) {
    std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
    OperatorBase* local_op_ptr = local_op.release();
    int op_role = boost::get<int>(
        op_desc->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));
    if (op_role == static_cast<int>(framework::OpRole::kOptimize)) {
      optimize_ops_.push_back(local_op_ptr);
      VLOG(3) << "append optimizer op " << local_op_ptr->Type();
    } else {
      forward_backward_ops_.push_back(local_op_ptr);
      VLOG(3) << "append forward/backward op " << local_op_ptr->Type();
    }
  }
  VLOG(3) << "create operators done.";
}

void MirroredWorker::CreateThreadScope(const ProgramDesc& program) {
  auto& block = program.Block(0);

  PADDLE_ENFORCE_NOT_NULL(
      root_scope_, "root_scope should be set before creating thread scope");

  VLOG(3) << "Going to create new scope";
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
  VLOG(3) << "end of create thread scope";
}

void MirroredWorker::BindingDataFeedMemory() {
  const std::vector<std::string>& input_feed =
      device_reader_->GetUseSlotAlias();
  for (auto name : input_feed) {
    device_reader_->AddFeedVar(thread_scope_->Var(name), name);
  }
}

void MirroredWorker::CreateDeviceResource(const ProgramDesc& main_prog) {
  CreateThreadScope(main_prog);
  CreateThreadOperators(main_prog);
}

void MirroredWorker::TrainFilesWithProfiler() {
  platform::SetNumThreads(1);
  device_reader_->Start();
  std::vector<double> forward_backward_op_time;
  std::vector<std::string> forward_backward_op_name;
  for (auto& op : forward_backward_ops_) {
    forward_backward_op_name.push_back(op->Type());
  }
  forward_backward_op_time.resize(forward_backward_op_name.size());
  for (size_t i = 0; i < forward_backward_op_time.size(); ++i) {
    forward_backward_op_time[i] = 0.0;
  }

  std::vector<double> all_reduce_time;
  all_reduce_time.resize(grad_names_.size());
  for (size_t i = 0; i < all_reduce_time.size(); ++i) {
    all_reduce_time[i] = 0.0;
  }

  std::vector<double> optimize_op_time;
  std::vector<std::string> optimize_op_name;
  for (auto& op : optimize_ops_) {
    optimize_op_name.push_back(op->Type());
  }
  optimize_op_time.resize(optimize_op_name.size());
  for (size_t i = 0; i < optimize_op_name.size(); ++i) {
    optimize_op_time[i] = 0.0;
  }

  platform::Timer timeline;
  double total_time = 0.0;
  double read_time = 0.0;
  int cur_batch;
  int batch_cnt = 0;
  timeline.Start();
  uint64_t total_inst = 0;
  while ((cur_batch = device_reader_->Next()) > 0) {
    timeline.Pause();
    read_time += timeline.ElapsedSec();
    for (size_t i = 0; i < forward_backward_ops_.size(); ++i) {
      bool need_skip = false;
      for (auto t = 0u; t < skip_ops_.size(); ++t) {
        if (forward_backward_ops_[i]->Type().find(skip_ops_[t]) !=
            std::string::npos) {
          need_skip = true;
          break;
        }
      }
      timeline.Start();
      VLOG(3) << "Going to run op " << forward_backward_op_name[i];
      if (!need_skip) {
        forward_backward_ops_[i]->Run(*thread_scope_, place_);
      }
      timeline.Pause();
      forward_backward_op_time[i] += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
    }
    total_inst += cur_batch;
    ++batch_cnt;
    PrintFetchVars();

    for (size_t i = 0; i < grad_names_.size(); ++i) {
      timeline.Start();
      nccl_ptr_->AllReduce(*thread_scope_, grad_names_[i], grad_names_[i],
                           place_);
      timeline.Pause();
      all_reduce_time[i] += timeline.ElapsedSec();
    }

    for (size_t i = 0; i < optimize_ops_.size(); ++i) {
      timeline.Start();
      optimize_ops_[i]->Run(*thread_scope_, place_);
      timeline.Pause();
      optimize_op_time[i] += timeline.ElapsedSec();
    }

    PrintFetchVars();
    if (thread_id_ == 0) {
      if (batch_cnt > 0 && batch_cnt % 100 == 0) {
        for (size_t i = 0; i < forward_backward_ops_.size(); ++i) {
          fprintf(stderr, "op_name:[%zu][%s], op_mean_time:[%fs]\n", i,
                  forward_backward_op_name[i].c_str(),
                  forward_backward_op_time[i] / batch_cnt);
        }
        for (size_t i = 0; i < optimize_ops_.size(); ++i) {
          fprintf(stderr, "optimize_ops_name:[%zu][%s], op_mean_time:[%fs]\n",
                  i, optimize_op_name[i].c_str(),
                  optimize_op_time[i] / batch_cnt);
        }
        for (size_t i = 0; i < grad_names_.size(); ++i) {
          fprintf(stderr, "all reduce time, var_name[%s], mean_time:[%fs]\n",
                  grad_names_[i].c_str(), all_reduce_time[i] / batch_cnt);
        }
      }
    }
    thread_scope_->DropKids();
    timeline.Start();
  }
}

void MirroredWorker::TrainFiles() {
  platform::SetNumThreads(1);

  // how to accumulate fetched values here
  device_reader_->Start();
  int cur_batch;
  VLOG(3) << "begin to train on device " << device_id_;

  while ((cur_batch = device_reader_->Next()) > 0) {
    for (auto& op : forward_backward_ops_) {
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

    for (auto& grad_name : grad_names_) {
      nccl_ptr_->AllReduce(*thread_scope_, grad_name, grad_name, place_);
    }

    for (auto& op : optimize_ops_) {
      op->Run(*thread_scope_, place_);
    }

    PrintFetchVars();
    thread_scope_->DropKids();
  }
}

void MirroredWorker::PrintFetchVars() {
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
