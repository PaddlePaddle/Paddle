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

#include <set>
#include <unordered_map>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/lodtensor_printer.h"

bool HasDependentOutput(
        const OpDesc& op_desc,
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

bool HasDependentInput(
        const OpDesc& op_desc,
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

bool NotHasDependentOutput(
        const OpDesc& op_desc,
        const std::unordered_set<std::string>& dependent_vars) {
    for (auto& var : op_desc.Outputs()) {
        for (auto& argu : var.second) {
            if (dependent_vars.count(argu) == 0) {
                return true;
            }
        }
    }
    return false;
}

bool HasOutput(
        const OpDesc& op_desc,
        const std::string& name) {
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
void DownpourWorkerOpt::CreateThreadOperatorsWithRerank(const ProgramDesc& program) {
    auto &block = program.Block(0);
    auto* ops = block.AllOps();
    // check if Independent between losses if not skip for now
    int loss_num = loss_names_.size();
    std::unordered_map<std::string, std::unordered_set<std::string>> loss_input_map; 
    std::unordered_map<std::string, std::unordered_set<std::string>> loss_output_map;
    // mark forward ops by loss
    for (int i = 0; i < loss_num; i++) {
      for (auto op_iter = ops.rbegin(); op_iter != ops.rend(); ++op_iter) {
        auto &op_desc = *op_iter;
        if (i > 0) {
          for (int j = 0; j < i; j++) {
            if (HasDependentInput(*op_desc, loss_input_map[loss_names[j]])) {
              VLOG(3) << "losses must be independence currently";
              return;
            }
          }
        }
        if (HasOutput(*op_desc, loss_names[i]) || HasDependentOutput(*op_desc, loss_input_map[loss_names[i]])) {
          AppendInputVar(*op_desc, &loss_input_map[loss_names[i]]);
          AppendOutputVar(*op_desc, &loss_output_map[loss_names[i]]);
        }
      }
    }


    //
    std::vector<std::string> loss_grad_names;
    for (int i = 0; i < loss_num; i++) {
        loss_grad_names.push_back(loss_names_[i]+"@GRAD");
    }
    std::unordered_map<std::string, std::unordered_set<std::string>> loss_input_grad_map;
    std::unordered_map<std::string, std::unordered_set<std::string>> loss_output_grad_map;
    for (int i = 0; i < loss_num; i++) {
        for (auto op_iter = ops.begin(); op_iter != ops.end(); ++op_iter) {
            auto &op_desc = *op_iter;

            if (HasOutput(*op_desc, loss_grad_names[i]) || 
                HasDependentOutput(*op_desc, loss_output_grad_map[loss_names_[i]])) {
                AppendInputVar(*op_desc, &loss_input_grad_map[loss_names_[i]]);
                AppendOutputVar(*op_desc, &loss_output_grad_map[loss_names_[i]]);
            }
        }
    }


    std::unordered_map<std::string, std::unordered_set<std::string>> metric_input_map;
    std::unordered_map<std::string, std::unordered_set<std::string>> metric_output_map;
    for (int i = 0; i < loss_num; i++) {
        for (auto op_iter = ops.begin(); op_iter != ops.end(); ++op_iter) {
            auto &op_desc = *op_iter;
            if ((HasDependentInput(*op_desc, loss_output_map[loss_names_[i]]) &&
                NotHasDependentOutput(op_desc, loss_input_map[loss_name_[i]])) ||
                HasDependentInput(op_desc, metric_output_vars[loss_names_[i]])) {
                AppendInputVar(op_desc, &metric_input_map[loss_names_[i]]);
                AppendOutputVar(op_desc, &metric_input_vars[loss_names_[i]]);
            }
        }
    }
    
    for (int i = 0; i < param_.program_config(0).pull_sparse_table_id_size();
         ++i) {
      uint64_t tid = static_cast<uint64_t>(
          param_.program_config(0).pull_sparse_table_id(i));
      TableParameter table;
      for (auto j : param_.sparse_table()) {
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
    for (int i = 0; i < loss_num; i++) {
        for (auto op_iter = ops->begin(); op_iter != ops->end(); ++op_iter) {
            auto &op_desc = *op_iter;
            if (HasDependentInput(op_desc, loss_input_vars[i]) ||
                HasDependentInput(op_desc, loss_input_grad_vars[i]) ||
                HasDependentInput(op_desc, metric_input_vars[i])) {
                std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
                if (HasOutput(*op_desc, async_wait_name_)) {
                  loss_op_names_[i].push_back("async_wait_flag");    
                } else {
                  loss_op_names_[i].push_back(op_desc->Type());
                }
                OperatorBase *local_op_ptr = local_op.release();
                loss_ops_[i].push_back(local_op_ptr);
            }
        }
    }
}

void DownpourWorker::TrainFiles() {
  VLOG(3) << "Begin to train files";
  platform::SetNumThreads(1);
  device_reader_->Start();
  int batch_cnt = 0;
  int cur_batch;
  std::future<int32_t> pull_async_status;
  std::string async_wait_name = "";
  for (int i = 0; i < param_.program_config(0).pull_sparse_table_id_size();
         ++i) {
      uint64_t tid = static_cast<uint64_t>(
          param_.program_config(0).pull_sparse_table_id(i));
      TableParameter table;
      for (auto j : param_.sparse_table()) {
        if (j.table_id() == tid) {
          table = j;
          break;
        }
      }
  }
  // pre-defined for the first op run with async-pulled embedding
  // uint64_t general_tid = 1;
  while ((cur_batch = device_reader_->Next()) > 0) {
    if (copy_table_config_.need_copy()) {
      if (copy_table_config_.sparse_copy_by_feasign()) {
        for (size_t i = 0; i < copy_sparse_tables_.size(); ++i) {
          uint64_t tid = copy_sparse_tables_[i].first;
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
      for (auto j : param_.sparse_table()) {
        if (j.table_id() == tid) {
          table = j;
          break;
        }
      }
      if (table.is_local()) {
        fleet_ptr_->PullSparseVarsFromLocal(*thread_scope_, tid,
                                       sparse_key_names_[tid], &features_[tid],
                                       &feature_values_[tid], table.fea_dim());
        // FillSparseFromLocal(*thread_scope_, tid, sparse_key_names_[tid], fleet_ptr_->GetLocalTable());
        // std::cout << "local sparse table with fea dim: " << table.fea_dim() << std::endl;
        CollectLabelInfo(i);
        // general_tid = tid;
        continue;
      } else if (table.is_async()) {
        pull_async_status = fleet_ptr_->PullSparseVarsAsync(*thread_scope_, tid,
                                            sparse_key_names_[tid], &features_[tid],
                                            &feature_values_[tid], table.fea_dim());
        continue;
      } else {
        fleet_ptr_->PullSparseVarsSync(
              *thread_scope_, tid, sparse_key_names_[tid], &features_[tid],
              &feature_values_[tid], table.fea_dim(), sparse_value_names_[tid]); 
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
    for (int loss_idx = 0; loss_idx < loss_ops_.szie(); loss_idx++) {
      int op_idx = 0;  
      for (auto& op : loss_ops_[loss_idx]) {
        bool need_skip = false;
        for (auto t = 0u; t < skip_ops_.size(); ++t) {
          if (op->Type().find(skip_ops_[t]) != std::string::npos) {
            need_skip = true;
            break;
          }
        }
        if (!need_skip) {
          if (loss_op_names[loss_idx][op_idx] == async_wait_name_) {
          // std::cout << "Wait Async Pull with tid: " << async_tid << std::endl;
             pull_async_status.wait();
            auto status = pull_async_status.get();
            if (status != 0) {
              LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
              sleep(1);
              exit(-1);
            } else {
              // std::cout << "Done Async Pull with tid: " << async_tid << std::endl;  
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
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      if (tensor == nullptr) {
        continue;
      }
      PADDLE_ENFORCE_EQ(framework::TensorContainsInf(*tensor), false,
                        "Tensor %s contains Inf", var_name);
      PADDLE_ENFORCE_EQ(framework::TensorContainsNAN(*tensor), false,
                        "Tensor %s contains NAN", var_name);
    }

    if (need_to_push_sparse_) {
      // push gradients here
      for (int i = 0; i < param_.program_config(0).push_sparse_table_id_size();
           ++i) {
        uint64_t tid = static_cast<uint64_t>(
            param_.program_config(0).push_sparse_table_id(i));
        TableParameter table;
        for (auto i : param_.sparse_table()) {
          if (i.table_id() == tid) {
            table = i;
            break;
          }
        }
        fleet_ptr_->PushSparseVarsWithLabelAsync(
            *thread_scope_, tid, features_[tid], feature_labels_[tid],
            sparse_key_names_[tid], sparse_grad_names_[tid], table.emb_dim(),
            &feature_grads_[tid], &push_sparse_status_, cur_batch, use_cvm_,
            dump_slot_, &sparse_push_keys_[tid], no_cvm_);
      }
    }

    if (need_to_push_dense_) {
      for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
           ++i) {
        uint64_t tid = static_cast<uint64_t>(
            param_.program_config(0).push_dense_table_id(i));
        fleet_ptr_->PushDenseVarsAsync(
            *thread_scope_, tid, dense_grad_names_[tid], &push_sparse_status_,
            scale_datanorm_, cur_batch);
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
      size_t batch_size = device_reader_->GetCurBatchSize();
      std::vector<std::string> ars(batch_size);
      for (auto& ar : ars) {
        ar.clear();
      }
      auto& ins_id_vec = device_reader_->GetInsIdVec();
      auto& ins_content_vec = device_reader_->GetInsContentVec();
      for (size_t i = 0; i < ins_id_vec.size(); i++) {
        ars[i] += ins_id_vec[i];
        ars[i] = ars[i] + "\t" + ins_content_vec[i];
      }
      for (auto& field : dump_fields_) {
        Variable* var = thread_scope_->FindVar(field);
        if (var == nullptr) {
          continue;
        }
        LoDTensor* tensor = var->GetMutable<LoDTensor>();
        if (!CheckValidOutput(tensor, batch_size)) {
          continue;
        }
        for (size_t i = 0; i < batch_size; ++i) {
          auto output_dim = tensor->dims()[1];
          std::string output_dimstr =
              boost::lexical_cast<std::string>(output_dim);
          ars[i] = ars[i] + "\t" + field + ":" + output_dimstr;
          auto bound = GetTensorBound(tensor, i);
          ars[i] += PrintLodTensor(tensor, bound.first, bound.second);
        }
      }
      // #pragma omp parallel for
      for (size_t i = 0; i < ars.size(); i++) {
        if (ars[i].length() == 0) {
          continue;
        }
        writer_ << ars[i];
      }
      if (need_dump_param_ && thread_id_ == 0) {
        DumpParam();
      }
    }

    PrintFetchVars();
    thread_scope_->DropKids();
    ++batch_cnt;
  }
  if (need_dump_field_) {
    writer_.Flush();
  }
  if (copy_table_config_.need_copy()) {
    CopySparseTable();
    CopyDenseTable();
    CopyDenseVars();
  }
}
