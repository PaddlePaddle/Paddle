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

#if defined(PADDLE_WITH_PSCORE)
#include "paddle/common/enforce.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/fleet/metrics.h"
#include "paddle/fluid/operators/isfinite_op.h"
#include "paddle/phi/core/platform/cpu_helper.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle::framework {
class Variable;
}  // namespace paddle::framework

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

namespace paddle::framework {
void DownpourLiteWorker::Initialize(const TrainerDesc& desc) {
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

  flag_partial_push_ = false;
  for (auto& m : param_.program_config(0).partial_pushdense_condtable_map()) {
    cond2table_map_[m.key()] = m.value();
    condvalue_set_.insert(m.value());
    flag_partial_push_ = true;
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

  fleet_ptr_ = paddle::distributed::FleetWrapper::GetInstance();
  fetch_config_ = desc.fetch_config();
  use_cvm_ = desc.use_cvm();
  // for sparse value accessor, embedding only
  no_cvm_ = desc.no_cvm();
  scale_sparse_gradient_with_batch_size_ =
      desc.scale_sparse_gradient_with_batch_size();
  scale_datanorm_ = desc.scale_datanorm();
  dump_slot_ = desc.dump_slot();
  adjust_ins_weight_config_ = desc.adjust_ins_weight_config();
  for (int i = 0; i < desc.check_nan_var_names_size(); ++i) {
    check_nan_var_names_.push_back(desc.check_nan_var_names(i));
  }
  copy_table_config_ = desc.copy_table_config();
  for (int i = 0; i < copy_table_config_.src_sparse_tables_size(); ++i) {
    uint64_t src_table = copy_table_config_.src_sparse_tables(i);
    uint64_t dest_table = copy_table_config_.dest_sparse_tables(i);
    VLOG(3) << "copy_sparse_tables_ push back " << src_table << "->"
            << dest_table;
    copy_sparse_tables_.push_back(std::make_pair(src_table, dest_table));
  }
  for (int i = 0; i < copy_table_config_.src_dense_tables_size(); ++i) {
    uint64_t src_table = copy_table_config_.src_dense_tables(i);
    uint64_t dest_table = copy_table_config_.dest_dense_tables(i);
    VLOG(3) << "copy_dense_tables_ push back " << src_table << "->"
            << dest_table;
    copy_dense_tables_.push_back(std::make_pair(src_table, dest_table));
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

void DownpourLiteWorker::CopySparseTable() {
  for (auto& copy_sparse_table : copy_sparse_tables_) {
    int64_t src_table = copy_sparse_table.first;
    int64_t dest_table = copy_sparse_table.second;
    int32_t feanum = 0;
    if (src_table == dest_table) {
      continue;
    } else if (!copy_table_config_.sparse_copy_by_feasign()) {
      if (feasign_set_.find(src_table) == feasign_set_.end()) {
        continue;
      } else if (feasign_set_[src_table].empty()) {
        continue;
      }
      feanum = fleet_ptr_->CopyTable(src_table, dest_table);
    } else {
      std::vector<uint64_t> fea_vec(feasign_set_[src_table].begin(),
                                    feasign_set_[src_table].end());
      feanum = fleet_ptr_->CopyTableByFeasign(src_table, dest_table, fea_vec);
      fea_vec.clear();
      std::vector<uint64_t>().swap(fea_vec);
    }
    VLOG(3) << "copy feasign from table " << src_table << " to table "
            << dest_table << ", feasign num=" << feanum;
    feasign_set_[src_table].clear();
    std::unordered_set<uint64_t>().swap(feasign_set_[src_table]);
  }
  feasign_set_.clear();
}

void DownpourLiteWorker::CopyDenseTable() {
  if (thread_id_ != 0) {
    return;
  }
  thread_local std::vector<std::future<int32_t>> pull_dense_status;
  for (auto& copy_dense_table : copy_dense_tables_) {
    uint64_t src_table = copy_dense_table.first;
    uint64_t dest_table = copy_dense_table.second;
    if (src_table == dest_table) {
      continue;
    }
    int32_t dim = fleet_ptr_->CopyTable(src_table, dest_table);
    VLOG(3) << "copy param from table " << src_table << " to table "
            << dest_table << ", dim=" << dim;
    if (copy_table_config_.dense_pull_after_copy()) {
      VLOG(3) << "dense pull after copy, table=" << dest_table;
      pull_dense_status.resize(0);
      fleet_ptr_->PullDenseVarsAsync(*root_scope_,
                                     dest_table,
                                     dense_value_names_[dest_table],
                                     &pull_dense_status,
                                     true);
      for (auto& t : pull_dense_status) {
        t.wait();
        auto status = t.get();
        if (status != 0) {
          LOG(WARNING) << "pull dense after copy table failed,"
                       << " table=" << dest_table;
        }
      }
    }
  }
}

void DownpourLiteWorker::CopyDenseVars() {
  if (thread_id_ != 0) {
    return;
  }
  for (int i = 0; i < copy_table_config_.src_var_list_size(); ++i) {
    auto& src_var_name = copy_table_config_.src_var_list(i);
    auto& dest_var_name = copy_table_config_.dest_var_list(i);
    if (src_var_name == dest_var_name) {
      continue;
    }
    VLOG(3) << "copy dense var from " << src_var_name << " to "
            << dest_var_name;
    Variable* src_var = thread_scope_->FindVar(src_var_name);
    PADDLE_ENFORCE_EQ(src_var != nullptr,
                      true,
                      common::errors::NotFound("%s not found", src_var_name));
    phi::DenseTensor* src_tensor = src_var->GetMutable<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(
        src_tensor != nullptr,
        true,
        common::errors::InvalidArgument("%s tensor is null", src_var_name));
    float* src_data = src_tensor->data<float>();

    Variable* dest_var = thread_scope_->FindVar(dest_var_name);
    PADDLE_ENFORCE_EQ(dest_var != nullptr,
                      true,
                      common::errors::NotFound("%s not found", dest_var_name));
    phi::DenseTensor* dest_tensor = dest_var->GetMutable<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(
        dest_tensor != nullptr,
        true,
        common::errors::InvalidArgument("%s tensor is null", dest_var_name));
    float* dest_data = dest_tensor->data<float>();

    PADDLE_ENFORCE_EQ(
        src_tensor->numel(),
        dest_tensor->numel(),
        common::errors::InvalidArgument("tensor numel not equal, %d vs %d",
                                        src_tensor->numel(),
                                        dest_tensor->numel()));
    for (int i = 0; i < src_tensor->numel(); i++) {
      dest_data[i] = src_data[i];
    }
  }
}

void DownpourLiteWorker::TrainFilesWithProfiler() {
  VLOG(3) << "Begin to train files with profiler";
  platform::SetNumThreads(1);
  device_reader_->Start();
  std::vector<double> op_total_time;
  std::vector<std::string> op_name;
  for (auto& op : ops_) {
    bool need_skip = false;
    for (auto& skip_op : skip_ops_) {
      if (op->Type().find(skip_op) != std::string::npos) {
        need_skip = true;
        break;
      }
    }
    if (!need_skip) {
      op_name.push_back(op->Type());
    }
  }

  VLOG(3) << "op name size: " << op_name.size();
  op_total_time.resize(op_name.size());
  for (double& op_time : op_total_time) {
    op_time = 0.0;
  }
  platform::Timer timeline;
  double total_time = 0.0;
  double read_time = 0.0;
  double pull_sparse_time = 0.0;
  double adjust_ins_weight_time = 0.0;
  double collect_label_time = 0.0;
  double fill_sparse_time = 0.0;
  double push_sparse_time = 0.0;
  double push_dense_time = 0.0;
  double copy_table_time = 0.0;
  int cur_batch;
  int batch_cnt = 0;
  uint64_t total_inst = 0;
  timeline.Start();
  while ((cur_batch = device_reader_->Next()) > 0) {
    timeline.Pause();
    read_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();

    timeline.Start();
    if (copy_table_config_.need_copy()) {
      VLOG(3) << "copy_sparse_tables_.size " << copy_sparse_tables_.size();
      if (batch_cnt % copy_table_config_.batch_num() == 0) {
        CopySparseTable();
        CopyDenseTable();
        CopyDenseVars();
      }
    }
    timeline.Pause();
    copy_table_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();

    int run_op_idx = 0;
    for (auto& op : ops_) {
      bool need_skip = false;
      for (auto& skip_op : skip_ops_) {
        if (op->Type().find(skip_op) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      if (!need_skip) {
        timeline.Start();
        VLOG(3) << "Going to run op " << op_name[run_op_idx];
        op->Run(*thread_scope_, place_);
        VLOG(3) << "Op " << op_name[run_op_idx] << " Finished";
        timeline.Pause();
        op_total_time[run_op_idx++] += timeline.ElapsedSec();
        total_time += timeline.ElapsedSec();
      }
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
      PADDLE_ENFORCE_EQ(framework::TensorContainsInf(*tensor),
                        false,
                        common::errors::InvalidArgument(
                            "phi::DenseTensor %s contains Inf.", var_name));
      PADDLE_ENFORCE_EQ(framework::TensorContainsNAN(*tensor),
                        false,
                        common::errors::InvalidArgument(
                            "phi::DenseTensor %s contains NAN.", var_name));
    }

#if defined(PADDLE_WITH_PSLIB) || defined(PADDLE_WITH_PSCORE)
    if (copy_table_config_.need_copy()) {
      if (copy_table_config_.sparse_copy_by_feasign()) {
        for (auto& copy_sparse_table : copy_sparse_tables_) {
          uint64_t tid = copy_sparse_table.first;
          feasign_set_[tid].insert(sparse_push_keys_[tid].begin(),
                                   sparse_push_keys_[tid].end());
        }
      }
    }
#endif

    if (need_to_push_dense_) {
      for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
           ++i) {
        uint64_t tid = static_cast<uint64_t>(
            param_.program_config(0).push_dense_table_id(i));
        pull_dense_worker_->IncreaseThreadVersion(thread_id_, tid);
      }
    }

    PrintFetchVars();
    thread_scope_->DropKids();
    total_inst += cur_batch;
    ++batch_cnt;

    if (thread_id_ == 0) {
      // should be configured here
      if (batch_cnt > 0 && batch_cnt % 100 == 0) {
        double op_sum_time = 0;
        std::unordered_map<std::string, double> op_to_time;
        for (size_t i = 0; i < op_total_time.size(); ++i) {
          fprintf(stderr,
                  "op_name:[%zu][%s], op_mean_time:[%fs]\n",
                  i,
                  op_name[i].c_str(),
                  op_total_time[i] / batch_cnt);
          if (op_to_time.find(op_name[i]) == op_to_time.end()) {
            op_to_time[op_name[i]] = 0.0;
          }
          op_to_time[op_name[i]] += op_total_time[i];
          op_sum_time += op_total_time[i];
        }
        for (auto& i : op_to_time) {
          fprintf(stderr,
                  "op [%s] run total time: [%f]ms\n",
                  i.first.c_str(),
                  i.second / batch_cnt);
        }
        fprintf(stderr, "op run total time: %fs\n", op_sum_time / batch_cnt);
        fprintf(stderr, "train total time: %fs\n", total_time / batch_cnt);
        fprintf(
            stderr, "pull sparse time: %fs\n", pull_sparse_time / batch_cnt);
        fprintf(
            stderr, "fill sparse time: %fs\n", fill_sparse_time / batch_cnt);
        fprintf(
            stderr, "push sparse time: %fs\n", push_sparse_time / batch_cnt);
        fprintf(stderr, "push dense time: %fs\n", push_dense_time / batch_cnt);
        fprintf(stderr,
                "collect label time: %fs\n",
                collect_label_time / batch_cnt);
        fprintf(stderr,
                "adjust ins weight time: %fs\n",
                adjust_ins_weight_time / batch_cnt);
        fprintf(stderr, "copy table time: %fs\n", copy_table_time / batch_cnt);
        fprintf(stderr, "mean read time: %fs\n", read_time / batch_cnt);
        fprintf(stderr, "IO percent: %f\n", read_time / total_time * 100);
        fprintf(stderr, "op run percent: %f\n", op_sum_time / total_time * 100);
        fprintf(stderr,
                "pull sparse time percent: %f\n",
                pull_sparse_time / total_time * 100);
        fprintf(stderr,
                "adjust ins weight time percent: %f\n",
                adjust_ins_weight_time / total_time * 100);
        fprintf(stderr,
                "copy table time percent: %f\n",
                copy_table_time / total_time * 100);
        fprintf(stderr,
                "collect label time percent: %f\n",
                collect_label_time / total_time * 100);
        fprintf(stderr,
                "fill sparse time percent: %f\n",
                fill_sparse_time / total_time * 100);
        fprintf(stderr,
                "push sparse time percent: %f\n",
                push_sparse_time / total_time * 100);
        fprintf(stderr,
                "push dense time percent: %f\n",
                push_dense_time / total_time * 100);
        fprintf(
            stderr, "%6.2f instances/s\n", total_inst / total_time);  // NOLINT
      }
    }
    timeline.Start();
  }
  if (copy_table_config_.need_copy()) {
    CopySparseTable();
    CopyDenseTable();
    CopyDenseVars();
  }
}

#if defined(PADDLE_WITH_PSLIB) || defined(PADDLE_WITH_PSCORE)
/**
 * @brief add auc monitor
 */
inline void AddAucMonitor(const Scope* scope, const phi::Place& place) {
  auto metric_ptr = Metric::GetInstance();
  auto& metric_list = metric_ptr->GetMetricList();
  for (auto& metric_item : metric_list) {
    auto* metric_msg = metric_item.second;
    if (metric_ptr->Phase() != metric_msg->MetricPhase()) {
      continue;
    }
    metric_msg->add_data(scope, place);
  }
}
#endif

void DownpourLiteWorker::TrainFiles() {
  VLOG(3) << "Begin to train files";
  platform::SetNumThreads(1);
  device_reader_->Start();
  int batch_cnt = 0;
  int cur_batch;
  while ((cur_batch = device_reader_->Next()) > 0) {
    if (copy_table_config_.need_copy()) {
      VLOG(3) << "Begin to copy table";
      if (batch_cnt % copy_table_config_.batch_num() == 0) {
        CopySparseTable();
        CopyDenseTable();
        CopyDenseVars();
      }
    }

    // do computation here
    for (auto& op : ops_) {
      bool need_skip = false;
      for (auto& skip_op : skip_ops_) {
        if (op->Type().find(skip_op) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      if (!need_skip) {
#if defined(PADDLE_WITH_PSLIB) || defined(PADDLE_WITH_PSCORE)
        try {
          op->Run(*thread_scope_, place_);
        } catch (std::exception& e) {
          fprintf(stderr, "error message: %s\n", e.what());
          auto& ins_id_vec = device_reader_->GetInsIdVec();
          size_t batch_size = device_reader_->GetCurBatchSize();
          std::string s = "";
          for (auto& ins_id : ins_id_vec) {
            if (!s.empty()) s += ",";
            s += ins_id;
          }
          fprintf(stderr,
                  "batch_size: %zu, ins_ids_vec: %s\n",
                  batch_size,
                  s.c_str());
          s = "";
          for (auto& param : all_param_) {
            Variable* var = thread_scope_->FindVar(param);
            if (var == nullptr) {
              continue;
            }
            phi::DenseTensor* tensor = nullptr;
            int64_t len = 0;
            if (var->IsType<phi::DenseTensor>()) {
              tensor = var->GetMutable<phi::DenseTensor>();
              len = tensor->numel();
            } else if (var->IsType<phi::SelectedRows>()) {
              auto selected_rows = var->GetMutable<phi::SelectedRows>();
              tensor = selected_rows->mutable_value();
              len = tensor->numel();
            }
            if (!tensor->IsInitialized()) {
              continue;
            }
            s += param + ":" + std::to_string(len) + ":";
            s += PrintDenseTensor(tensor, 0, len);
            fprintf(stderr, "%s\n", s.c_str());
            fflush(stderr);
            s = "";
          }
          throw e;
        }
#else
        op->Run(*thread_scope_, place_);
#endif
      }
    }

#if defined(PADDLE_WITH_PSLIB) || defined(PADDLE_WITH_PSCORE)
    // add data for MetricMsg
    if (Metric::GetInstance() != nullptr) {
      AddAucMonitor(thread_scope_, place_);
    }
#endif

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
      PADDLE_ENFORCE_EQ(framework::TensorContainsInf(*tensor),
                        false,
                        common::errors::InvalidArgument(
                            "phi::DenseTensor %s contains Inf.", var_name));
      PADDLE_ENFORCE_EQ(framework::TensorContainsNAN(*tensor),
                        false,
                        common::errors::InvalidArgument(
                            "phi::DenseTensor %s contains NAN.", var_name));
    }

#if defined(PADDLE_WITH_PSLIB) || defined(PADDLE_WITH_PSCORE)
    if (copy_table_config_.need_copy()) {
      if (copy_table_config_.sparse_copy_by_feasign()) {
        for (auto& copy_sparse_table : copy_sparse_tables_) {
          uint64_t tid = copy_sparse_table.first;
          feasign_set_[tid].insert(sparse_push_keys_[tid].begin(),
                                   sparse_push_keys_[tid].end());
        }
      }
    }
#endif

    // TODO(zhaocaibei123): flag_partial_push_ => op

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
#endif
