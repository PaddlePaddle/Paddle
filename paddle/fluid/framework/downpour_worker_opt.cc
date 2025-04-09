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
#include "paddle/fluid/operators/isfinite_op.h"
#include "paddle/phi/core/platform/cpu_helper.h"

namespace paddle::framework {

class OpDesc;
class OperatorBase;
class ProgramDesc;

bool HasDependentOutput(const OpDesc& op_desc,
                        const std::unordered_set<std::string>& dependent_vars) {
  for (auto& var : op_desc.Outputs()) {
    for (auto& argu : var.second) {
      if (dependent_vars.count(argu) != 0) {
        return true;
      }
    }
  }
  return false;
}

bool HasDependentInput(const OpDesc& op_desc,
                       const std::unordered_set<std::string>& dependent_vars) {
  for (auto& var : op_desc.Inputs()) {
    for (auto& argu : var.second) {
      if (dependent_vars.count(argu) != 0) {
        return true;
      }
    }
  }
  return false;
}

bool OnlyHasDependentInput(
    const OpDesc& op_desc,
    const std::unordered_set<std::string>& dependent_vars) {
  for (auto& var : op_desc.Inputs()) {
    for (auto& argu : var.second) {
      if (dependent_vars.count(argu) == 0) {
        return false;
      }
    }
  }
  return true;
}

bool NotHasDependentOutput(
    const OpDesc& op_desc,
    const std::unordered_set<std::string>& dependent_vars) {
  for (auto& var : op_desc.Outputs()) {
    for (auto& argu : var.second) {
      if (dependent_vars.count(argu) != 0) {
        return false;
      }
    }
  }
  return true;
}

bool HasOutput(const OpDesc& op_desc, const std::string& name) {
  for (auto& var : op_desc.Outputs()) {
    for (auto& argu : var.second) {
      if (argu == name) {
        return true;
      }
    }
  }
  return false;
}
void AppendInputVar(const OpDesc& op_desc,
                    std::unordered_set<std::string>* vars_set) {
  for (auto& var : op_desc.Inputs()) {
    for (auto& arg : var.second) {
      vars_set->emplace(arg);
    }
  }
}

void AppendOutputVar(const OpDesc& op_desc,
                     std::unordered_set<std::string>* vars_set) {
  for (auto& var : op_desc.Outputs()) {
    for (auto& arg : var.second) {
      vars_set->emplace(arg);
    }
  }
}

void DownpourWorkerOpt::Initialize(const TrainerDesc& desc) {
  param_ = desc.downpour_param();
  for (int i = 0; i < param_.sparse_table_size(); ++i) {
    uint64_t table_id =
        static_cast<uint64_t>(param_.sparse_table(i).table_id());
    TableParameter table = param_.sparse_table(i);
    sparse_key_names_[table_id].resize(table.sparse_key_name_size());
    for (int j = 0; j < table.sparse_key_name_size(); ++j) {
      sparse_key_names_[table_id][j] = table.sparse_key_name(j);
    }
    sparse_value_names_[table_id].resize(table.sparse_value_name_size());
    for (int j = 0; j < table.sparse_value_name_size(); ++j) {
      sparse_value_names_[table_id][j] = table.sparse_value_name(j);
    }
    sparse_grad_names_[table_id].resize(table.sparse_grad_name_size());
    for (int j = 0; j < table.sparse_grad_name_size(); ++j) {
      sparse_grad_names_[table_id][j] = table.sparse_grad_name(j);
    }
    label_var_name_[table_id] = table.label_var_name();
    sparse_push_keys_[table_id] = std::vector<uint64_t>();
  }

  for (int i = 0; i < param_.dense_table_size(); ++i) {
    uint64_t table_id = static_cast<uint64_t>(param_.dense_table(i).table_id());
    auto table = param_.dense_table(i);
    dense_value_names_[table_id].resize(table.dense_value_name_size());
    for (int j = 0; j < table.dense_value_name_size(); ++j) {
      dense_value_names_[table_id][j] = table.dense_value_name(j);
    }
    dense_grad_names_[table_id].resize(table.dense_grad_name_size());
    for (int j = 0; j < table.dense_grad_name_size(); ++j) {
      dense_grad_names_[table_id][j] = table.dense_grad_name(j);
    }
  }

  skip_ops_.resize(param_.skip_ops_size());
  for (int i = 0; i < param_.skip_ops_size(); ++i) {
    skip_ops_[i] = param_.skip_ops(i);
  }

  for (int i = 0; i < param_.stat_var_names_size(); ++i) {
    stat_var_name_map_[param_.stat_var_names(i)] = 1;
  }

  need_to_push_sparse_ = param_.push_sparse();
  need_to_push_dense_ = param_.push_dense();

  fleet_ptr_ = FleetWrapper::GetInstance();
  fetch_config_ = desc.fetch_config();
  use_cvm_ = desc.use_cvm();
  // for sparse value accessor, embedding only
  no_cvm_ = desc.no_cvm();
  scale_datanorm_ = desc.scale_datanorm();
  dump_slot_ = desc.dump_slot();
  adjust_ins_weight_config_ = desc.adjust_ins_weight_config();
  for (int i = 0; i < desc.loss_names_size(); ++i) {
    loss_names_.push_back(desc.loss_names(i));
  }
  for (int i = 0; i < desc.check_nan_var_names_size(); ++i) {
    check_nan_var_names_.push_back(desc.check_nan_var_names(i));
  }
  copy_table_config_ = desc.copy_table_config();
  for (int i = 0; i < copy_table_config_.src_sparse_tables_size(); ++i) {
    uint64_t src_table = copy_table_config_.src_sparse_tables(i);
    uint64_t dest_table = copy_table_config_.dest_sparse_tables(i);
    VLOG(3) << "copy_sparse_tables_ push back " << src_table << "->"
            << dest_table;
    copy_sparse_tables_.emplace_back(src_table, dest_table);
  }
  for (int i = 0; i < copy_table_config_.src_dense_tables_size(); ++i) {
    uint64_t src_table = copy_table_config_.src_dense_tables(i);
    uint64_t dest_table = copy_table_config_.dest_dense_tables(i);
    VLOG(3) << "copy_dense_tables_ push back " << src_table << "->"
            << dest_table;
    copy_dense_tables_.emplace_back(src_table, dest_table);
  }
  for (auto& m : copy_table_config_.table_dependency_map()) {
    if (sparse_key_names_.find(m.key()) != sparse_key_names_.end()) {
      // currently only support one dependency
      for (auto& value : m.values()) {
        table_dependency_[m.key()] = value;
      }
    }
  }
}

void DownpourWorkerOpt::CreateDeviceResource(const ProgramDesc& main_prog) {
  CreateThreadScope(main_prog);
  CreateThreadOperatorsWithRerank(main_prog);
}

void DownpourWorkerOpt::CreateThreadOperatorsWithRerank(
    const ProgramDesc& program) {
  auto& block = program.Block(0);
  std::vector<OpDesc*> ops = block.AllOps();
  // check if Independent between losses if not skip for now
  int loss_num = static_cast<int>(loss_names_.size());
  std::unordered_map<std::string, std::unordered_set<std::string>>
      loss_input_map;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      loss_output_map;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      loss_input_grad_map;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      loss_output_grad_map;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      metric_input_map;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      metric_output_map;
  std::vector<std::string> loss_grad_names;
  for (int i = 0; i < loss_num; i++) {
    loss_grad_names.push_back(loss_names_[i] + "@GRAD");
  }
  // mark forward ops by loss
  for (int i = 0; i < loss_num; i++) {
    for (auto op_iter = ops.rbegin(); op_iter != ops.rend(); ++op_iter) {
      auto& op_desc = *op_iter;
      if (i > 0) {
        for (int j = 0; j < i; j++) {
          if (HasDependentInput(*op_desc, loss_input_map[loss_names_[j]])) {
            VLOG(3) << "losses must be independence currently";
            return;
          }
        }
      }
      if (HasOutput(*op_desc, loss_names_[i]) ||
          HasOutput(*op_desc, loss_grad_names[i]) ||
          HasDependentOutput(*op_desc, loss_input_map[loss_names_[i]])) {
        AppendInputVar(*op_desc, &loss_input_map[loss_names_[i]]);
        AppendOutputVar(*op_desc, &loss_output_map[loss_names_[i]]);
      }
    }
  }

  for (int i = 0; i < loss_num; i++) {
    for (auto& op_desc : ops) {
      if (HasOutput(*op_desc, loss_grad_names[i]) ||
          HasDependentInput(*op_desc, loss_output_grad_map[loss_names_[i]])) {
        AppendInputVar(*op_desc, &loss_input_grad_map[loss_names_[i]]);
        AppendOutputVar(*op_desc, &loss_output_grad_map[loss_names_[i]]);
      }
    }
  }

  for (int i = 0; i < loss_num; i++) {
    for (auto& op_desc : ops) {
      if ((HasDependentInput(*op_desc, loss_output_map[loss_names_[i]]) &&
           OnlyHasDependentInput(*op_desc, loss_output_map[loss_names_[i]]) &&
           NotHasDependentOutput(*op_desc, loss_input_map[loss_names_[i]])) ||
          HasDependentInput(*op_desc, metric_output_map[loss_names_[i]])) {
        AppendInputVar(*op_desc, &metric_input_map[loss_names_[i]]);
        AppendOutputVar(*op_desc, &metric_output_map[loss_names_[i]]);
      }
    }
  }

  for (int i = 0; i < param_.program_config(0).pull_sparse_table_id_size();
       ++i) {
    uint64_t tid =
        static_cast<uint64_t>(param_.program_config(0).pull_sparse_table_id(i));
    TableParameter table;
    for (auto const& j : param_.sparse_table()) {
      if (j.table_id() == tid) {
        table = j;
        break;
      }
    }
    if (table.is_async()) {
      async_tid_ = tid;
      async_index_ = i;
      async_wait_name_ = table.async_wait_op_name();
    }
  }
  loss_op_names_.resize(loss_num);
  loss_ops_.resize(loss_num);
  std::string async_wait_flag = "async_wait_flag";
  for (int i = 0; i < loss_num; i++) {
    for (auto& op_desc : ops) {
      if ((op_desc->Type() == "fill_constant" &&
           HasDependentOutput(*op_desc,
                              loss_output_grad_map[loss_names_[i]])) ||
          (HasDependentInput(*op_desc, loss_input_map[loss_names_[i]]) &&
           HasDependentOutput(*op_desc, loss_output_map[loss_names_[i]])) ||
          (HasDependentInput(*op_desc, loss_input_grad_map[loss_names_[i]]) &&
           HasDependentOutput(*op_desc,
                              loss_output_grad_map[loss_names_[i]])) ||
          (HasDependentInput(*op_desc, metric_input_map[loss_names_[i]]) &&
           HasDependentOutput(*op_desc, metric_output_map[loss_names_[i]]))) {
        std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
        if (HasOutput(*op_desc, async_wait_name_)) {
          loss_op_names_[i].push_back(async_wait_flag);
        } else {
          loss_op_names_[i].push_back(op_desc->Type());
        }
        OperatorBase* local_op_ptr = local_op.release();
        loss_ops_[i].push_back(local_op_ptr);
      }
    }
  }
}

void DownpourWorkerOpt::TrainFiles() {
  VLOG(3) << "Begin to train files";
  platform::SetNumThreads(1);
  device_reader_->Start();
  int batch_cnt = 0;
  int cur_batch = 0;
  std::future<int32_t> pull_async_status;
  std::string async_wait_name = "";
  for (int i = 0; i < param_.program_config(0).pull_sparse_table_id_size();
       ++i) {
    uint64_t tid =
        static_cast<uint64_t>(param_.program_config(0).pull_sparse_table_id(i));
    TableParameter table;
    for (auto const& j : param_.sparse_table()) {
      if (j.table_id() == tid) {
        table = j;
        break;
      }
    }
  }
  // pre-defined for the first op run with async-pulled embedding
  while ((cur_batch = device_reader_->Next()) > 0) {
    if (copy_table_config_.need_copy()) {
      if (copy_table_config_.sparse_copy_by_feasign()) {
        for (auto& copy_sparse_table : copy_sparse_tables_) {
          uint64_t tid = copy_sparse_table.first;
          feasign_set_[tid].insert(sparse_push_keys_[tid].begin(),
                                   sparse_push_keys_[tid].end());
        }
      }
      if (batch_cnt % copy_table_config_.batch_num() == 0) {
        CopySparseTable();
        CopyDenseTable();
        CopyDenseVars();
      }
    }
    // pull sparse here
    for (int i = 0; i < param_.program_config(0).pull_sparse_table_id_size();
         ++i) {
      uint64_t tid = static_cast<uint64_t>(
          param_.program_config(0).pull_sparse_table_id(i));
      TableParameter table;
      for (auto const& j : param_.sparse_table()) {
        if (j.table_id() == tid) {
          table = j;
          break;
        }
      }
      if (table.is_local()) {
        fleet_ptr_->PullSparseVarsFromLocal(*thread_scope_,
                                            tid,
                                            sparse_key_names_[tid],
                                            &features_[tid],
                                            &feature_values_[tid],
                                            table.fea_dim());
        CollectLabelInfo(i);
        continue;
      } else if (table.is_async()) {
        pull_async_status =
            fleet_ptr_->PullSparseVarsAsync(*thread_scope_,
                                            tid,
                                            sparse_key_names_[tid],
                                            &features_[tid],
                                            &feature_values_[tid],
                                            table.fea_dim());
        continue;
      } else {
        fleet_ptr_->PullSparseVarsSync(*thread_scope_,
                                       tid,
                                       sparse_key_names_[tid],
                                       &features_[tid],
                                       &feature_values_[tid],
                                       table.fea_dim(),
                                       sparse_value_names_[tid]);
      }
      CollectLabelInfo(i);
      FillSparseValue(i);
      auto nid_iter = std::find(sparse_value_names_[tid].begin(),
                                sparse_value_names_[tid].end(),
                                adjust_ins_weight_config_.nid_slot());
      if (nid_iter != sparse_value_names_[tid].end()) {
        AdjustInsWeight();
      }
    }
    VLOG(3) << "fill sparse value for all sparse table done.";

    // do computation here
    for (size_t loss_idx = 0; loss_idx < loss_ops_.size(); loss_idx++) {
      int op_idx = 0;
      for (auto& op : loss_ops_[loss_idx]) {
        bool need_skip = false;
        for (auto& skip_op : skip_ops_) {
          if (op->Type().find(skip_op) != std::string::npos) {
            need_skip = true;
            break;
          }
        }
        if (!need_skip) {
          if (loss_op_names_[loss_idx][op_idx] == async_wait_name_) {
            pull_async_status.wait();
            auto status = pull_async_status.get();
            if (status != 0) {
              LOG(ERROR) << "fleet pull sparse failed, status[" << status
                         << "]";
              sleep(1);
              exit(-1);
            } else {
              // CollectLabelInfo(async_index);
              FillSparseValue(async_index_);
              auto nid_iter = std::find(sparse_value_names_[async_tid_].begin(),
                                        sparse_value_names_[async_tid_].end(),
                                        adjust_ins_weight_config_.nid_slot());
              if (nid_iter != sparse_value_names_[async_tid_].end()) {
                AdjustInsWeight();
              }
            }
          }
          op->Run(*thread_scope_, place_);
        }
      }
      op_idx++;
    }
    // check inf and nan
    for (std::string& var_name : check_nan_var_names_) {
      Variable* var = thread_scope_->FindVar(var_name);
      if (var == nullptr) {
        continue;
      }
      phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
      if (tensor == nullptr) {
        continue;
      }
      PADDLE_ENFORCE_EQ(
          framework::TensorContainsInf(*tensor),
          false,
          common::errors::InvalidArgument("The target tensor %s contains Inf "
                                          "should check some layers output.",
                                          var_name));
      PADDLE_ENFORCE_EQ(
          framework::TensorContainsNAN(*tensor),
          false,
          common::errors::InvalidArgument("The target tensor %s contains Nan "
                                          "should check some layers output.",
                                          var_name));
    }

    if (need_to_push_sparse_) {
      // push gradients here
      for (int i = 0; i < param_.program_config(0).push_sparse_table_id_size();
           ++i) {
        uint64_t tid = static_cast<uint64_t>(
            param_.program_config(0).push_sparse_table_id(i));
        TableParameter table;
        for (auto const& i : param_.sparse_table()) {
          if (i.table_id() == tid) {
            table = i;
            break;
          }
        }
        bool scale_sparse_gradient_with_batch_size_ = true;
        fleet_ptr_->PushSparseVarsWithLabelAsync(
            *thread_scope_,
            tid,
            features_[tid],
            feature_labels_[tid],
            sparse_key_names_[tid],
            sparse_grad_names_[tid],
            table.emb_dim(),
            &feature_grads_[tid],
            &push_sparse_status_,
            cur_batch,
            use_cvm_,
            dump_slot_,
            &sparse_push_keys_[tid],
            no_cvm_,
            scale_sparse_gradient_with_batch_size_);
      }
    }

    if (need_to_push_dense_) {
      for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
           ++i) {
        uint64_t tid = static_cast<uint64_t>(
            param_.program_config(0).push_dense_table_id(i));
        fleet_ptr_->PushDenseVarsAsync(*thread_scope_,
                                       tid,
                                       dense_grad_names_[tid],
                                       &push_sparse_status_,
                                       scale_datanorm_,
                                       cur_batch);
      }
      VLOG(3) << "push dense gradient done.";

      // the following code should be more precise and clean
      // TODO(guru4elephant)
      int32_t tmp_push_dense_wait_times = -1;
      static uint32_t push_dense_wait_times =
          static_cast<uint32_t>(tmp_push_dense_wait_times);

      if (push_dense_status_.size() >= push_dense_wait_times) {
        for (auto& t : push_dense_status_) {
          t.wait();
        }
        push_dense_status_.resize(0);
      }

      if (tmp_push_dense_wait_times == -1) {
        push_dense_status_.resize(0);
      }
    }

    if (need_to_push_sparse_) {
      VLOG(3) << "push sparse gradient done.";
      int32_t tmp_push_sparse_wait_times = -1;
      static uint32_t push_sparse_wait_times =
          static_cast<uint32_t>(tmp_push_sparse_wait_times);
      if (push_sparse_status_.size() >= push_sparse_wait_times) {
        for (auto& t : push_sparse_status_) {
          t.wait();
        }
        push_sparse_status_.resize(0);
      }

      if (tmp_push_sparse_wait_times == -1) {
        push_sparse_status_.resize(0);
      }
    }

    if (need_to_push_dense_) {
      for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
           ++i) {
        uint64_t tid = static_cast<uint64_t>(
            param_.program_config(0).push_dense_table_id(i));
        pull_dense_worker_->IncreaseThreadVersion(thread_id_, tid);
      }
    }
    if (need_dump_field_) {
      DumpField(*thread_scope_, dump_mode_, dump_interval_);
    }
    if (need_dump_param_ && thread_id_ == 0) {
      DumpParam(*thread_scope_, batch_cnt);
    }

    PrintFetchVars();
    thread_scope_->DropKids();
    ++batch_cnt;
  }
  if (need_dump_field_ || need_dump_param_) {
    writer_.Flush();
  }
  if (copy_table_config_.need_copy()) {
    CopySparseTable();
    CopyDenseTable();
    CopyDenseVars();
  }
}

}  // namespace paddle::framework
