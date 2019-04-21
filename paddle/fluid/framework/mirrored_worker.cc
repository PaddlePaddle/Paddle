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
  LOG(WARNING) << " nccl local rank: " << nccl_ptr_->nccl_info_.local_rank_
               << " global rank: " << nccl_ptr_->nccl_info_.my_global_rank_
               << " total ranks: " << nccl_ptr_->nccl_info_.global_ranks_;
#endif
}

void MirroredWorker::CreateThreadOperators(const ProgramDesc& program) {
  auto& block = program.Block(0);
  op_names_.clear();
  VLOG(3) << "begin to create operators";
  for (auto& op_desc : block.AllOps()) {
    std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
    op_names_.push_back(op_desc->Type());
    OperatorBase* local_op_ptr = local_op.release();
    forward_backward_ops_.push_back(local_op_ptr);
    int op_role = boost::get<int>(
        op_desc->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));
    if (op_role == static_cast<int>(framework::OpRole::kForward) ||
        op_role == static_cast<int>(framework::OpRole::kBackward) ||
        op_role == static_cast<int>(framework::OpRole::kLoss)) {
      forward_backward_ops_.push_back(local_op_ptr);
      VLOG(3) << "append forward/backward op " << local_op_ptr->Type();
    }
    if (op_role == static_cast<int>(framework::OpRole::kOptimize)) {
      optimize_ops_.push_back(local_op_ptr);
      VLOG(3) << "append optimizer op " << local_op_ptr->Type();
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

void MirroredWorker::TrainFiles() {
  platform::SetNumThreads(1);

  // how to accumulate fetched values here
  device_reader_->Start();
  int cur_batch;
  LOG(WARNING) << "begin to train on device " << device_id_;
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
