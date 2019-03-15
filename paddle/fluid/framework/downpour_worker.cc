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

namespace paddle {
namespace framework {

void DownpourWorker::Initialize(const TrainerDesc& desc) {
  param_ = desc.downpour_param();
  for (size_t i = 0; i < param_.sparse_table_size(); ++i) {
    uint64_t table_id =
        static_cast<uint64_t>(param_.sparse_table(i).table_id());
    TableParameter table = param_.sparse_table(i);
    sparse_key_names_[table_id].resize(table.sparse_key_name_size());
    for (size_t j = 0; j < table.sparse_key_name_size(); ++j) {
      sparse_key_names_[table_id][j] = table.sparse_key_name(j);
    }
    sparse_value_names_[table_id].resize(table.sparse_value_name_size());
    for (size_t j = 0; j < table.sparse_value_name_size(); ++j) {
      sparse_value_names_[table_id][j] = table.sparse_value_name(j);
    }
    sparse_grad_names_[table_id].resize(table.sparse_grad_name_size());
    for (size_t j = 0; j < table.sparse_grad_name_size(); ++j) {
      sparse_grad_names_[table_id][j] = table.sparse_grad_name(j);
    }
    label_var_name_[table_id] = table.label_var_name();
  }

  for (size_t i = 0; i < param_.dense_table_size(); ++i) {
    uint64_t table_id = static_cast<uint64_t>(param_.dense_table(i).table_id());
    auto table = param_.dense_table(i);
    dense_value_names_[table_id].resize(table.dense_value_name_size());
    for (size_t j = 0; j < table.dense_value_name_size(); ++j) {
      dense_value_names_[table_id][j] = table.dense_value_name(j);
    }
    dense_grad_names_[table_id].resize(table.dense_grad_name_size());
    for (size_t j = 0; j < table.dense_grad_name_size(); ++j) {
      dense_grad_names_[table_id][j] = table.dense_grad_name(j);
    }
  }

  skip_ops_.resize(param_.skip_ops_size());
  for (size_t i = 0; i < param_.skip_ops_size(); ++i) {
    skip_ops_[i] = param_.skip_ops(i);
  }

  fetch_var_names_.resize(desc.fetch_var_names_size());
  for (size_t i = 0; i < desc.fetch_var_names_size(); ++i) {
    fetch_var_names_[i] = desc.fetch_var_names(i);
  }

  batch_cnt_per_print_ = static_cast<int>(desc.batch_per_print());
  skip_ops_.resize(param_.skip_ops_size());
  fleet_ptr_ = FleetWrapper::GetInstance();
}

void DownpourWorker::CollectLabelInfo(size_t table_idx) {
  uint64_t table_id = static_cast<uint64_t>(
      param_.program_config(0).pull_sparse_table_id(table_idx));

  TableParameter table;
  for (auto i : param_.sparse_table()) {
    if (i.table_id() == table_id) {
      table = i;
      break;
    }
  }
  auto& feature = features_[table_id];
  auto& feature_label = feature_labels_[table_id];
  feature_label.resize(feature.size());
  Variable* var = thread_scope_->FindVar(label_var_name_[table_id]);
  LoDTensor* tensor = var->GetMutable<LoDTensor>();
  int64_t* label_ptr = tensor->data<int64_t>();

  int global_index = 0;
  for (size_t i = 0; i < sparse_key_names_[table_id].size(); ++i) {
    VLOG(3) << "sparse_key_names_[" << i
            << "]: " << sparse_key_names_[table_id][i];
    Variable* fea_var = thread_scope_->FindVar(sparse_key_names_[table_id][i]);
    LoDTensor* tensor = fea_var->GetMutable<LoDTensor>();
    int64_t* ids = tensor->data<int64_t>();
    int fea_idx = 0;
    // tensor->lod()[0].size() == batch_size + 1
    for (auto lod_idx = 1u; lod_idx < tensor->lod()[0].size(); ++lod_idx) {
      for (; fea_idx < tensor->lod()[0][lod_idx]; ++fea_idx) {
        // should be skipped feasign defined in protobuf
        if (ids[fea_idx] == 0u) {
          continue;
        }
        feature_label[global_index++] =
            static_cast<float>(label_ptr[lod_idx - 1]);
      }
    }
  }
  CHECK(global_index == feature.size())
      << "expect fea info size:" << feature.size() << " real:" << global_index;
}

void DownpourWorker::FillSparseValue(size_t table_idx) {
  uint64_t table_id = static_cast<uint64_t>(
      param_.program_config(0).pull_sparse_table_id(table_idx));

  TableParameter table;
  for (auto i : param_.sparse_table()) {
    if (i.table_id() == table_id) {
      table = i;
      break;
    }
  }

  auto& fea_value = feature_values_[table_id];
  auto fea_idx = 0u;

  std::vector<float> init_value(table.emb_dim());
  for (size_t i = 0; i < sparse_key_names_[table_id].size(); ++i) {
    std::string slot_name = sparse_key_names_[table_id][i];
    std::string emb_slot_name = sparse_value_names_[table_id][i];
    Variable* var = thread_scope_->FindVar(slot_name);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    int64_t* ids = tensor->data<int64_t>();
    int len = tensor->numel();
    Variable* var_emb = thread_scope_->FindVar(emb_slot_name);
    LoDTensor* tensor_emb = var_emb->GetMutable<LoDTensor>();
    float* ptr = tensor_emb->mutable_data<float>({len, table.emb_dim()},
                                                 platform::CPUPlace());
    memset(ptr, 0, sizeof(float) * len * table.emb_dim());
    auto& tensor_lod = tensor->lod()[0];
    LoD data_lod{tensor_lod};
    tensor_emb->set_lod(data_lod);
    for (auto index = 0u; index < len; ++index) {
      if (ids[index] == 0u) {
        memcpy(ptr + table.emb_dim() * index, init_value.data() + 2,
               sizeof(float) * table.emb_dim());
        continue;
      }
      memcpy(ptr + table.emb_dim() * index, fea_value[fea_idx].data() + 2,
             sizeof(float) * table.emb_dim());
      fea_idx++;
    }
  }
}

void DownpourWorker::TrainFilesWithProfiler() {
  VLOG(3) << "Begin to train files with profiler";
  platform::SetNumThreads(1);
  device_reader_->Start();
  std::vector<double> op_total_time;
  std::vector<std::string> op_name;
  for (auto& op : ops_) {
    bool need_skip = false;
    for (auto t = 0u; t < skip_ops_.size(); ++t) {
      if (op->Type().find(skip_ops_[t]) != std::string::npos) {
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
  for (size_t i = 0; i < op_total_time.size(); ++i) {
    op_total_time[i] = 0.0;
  }
  platform::Timer timeline;
  double total_time = 0.0;
  double read_time = 0.0;
  double pull_sparse_time = 0.0;
  double collect_label_time = 0.0;
  double fill_sparse_time = 0.0;
  double push_sparse_time = 0.0;
  double push_dense_time = 0.0;
  int cur_batch;
  int batch_cnt = 0;
  timeline.Start();
  while ((cur_batch = device_reader_->Next()) > 0) {
    timeline.Pause();
    read_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();
    VLOG(3) << "program config size: " << param_.program_config_size();
    for (size_t i = 0; i < param_.program_config(0).pull_sparse_table_id_size();
         ++i) {
      uint64_t tid = static_cast<uint64_t>(
          param_.program_config(0).pull_sparse_table_id(i));
      TableParameter table;
      for (auto i : param_.sparse_table()) {
        if (i.table_id() == tid) {
          table = i;
          break;
        }
      }
      timeline.Start();
      fleet_ptr_->PullSparseVarsSync(*thread_scope_, tid,
                                     sparse_key_names_[tid], &features_[tid],
                                     &feature_values_[tid], table.fea_dim());
      timeline.Pause();
      pull_sparse_time += timeline.ElapsedSec();
      CollectLabelInfo(i);
      timeline.Pause();
      collect_label_time += timeline.ElapsedSec();
      timeline.Start();
      FillSparseValue(i);
      timeline.Pause();
      fill_sparse_time += timeline.ElapsedSec();
    }
    VLOG(3) << "Fill sparse value for all sparse table done.";

    int run_op_idx = 0;
    for (auto& op : ops_) {
      bool need_skip = false;
      for (auto t = 0u; t < skip_ops_.size(); ++t) {
        if (op->Type().find(skip_ops_[t]) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      if (!need_skip) {
        timeline.Start();
        op->Run(*thread_scope_, place_);
        timeline.Pause();
        op_total_time[run_op_idx++] += timeline.ElapsedSec();
        total_time += timeline.ElapsedSec();
      }
    }

    for (size_t i = 0; i < param_.program_config(0).push_sparse_table_id_size();
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
      timeline.Start();
      fleet_ptr_->PushSparseVarsWithLabelAsync(
          *thread_scope_, tid, features_[tid], feature_labels_[tid],
          sparse_key_names_[tid], sparse_grad_names_[tid], table.emb_dim(),
          &feature_grads_[tid], &push_sparse_status_);
      timeline.Pause();
      push_sparse_time += timeline.ElapsedSec();
    }

    timeline.Start();
    for (size_t i = 0; i < param_.program_config(0).push_dense_table_id_size();
         ++i) {
      uint64_t tid = static_cast<uint64_t>(
          param_.program_config(0).push_dense_table_id(i));
      fleet_ptr_->PushDenseVarsAsync(
          *thread_scope_, tid, dense_grad_names_[tid], &push_sparse_status_);
    }
    timeline.Pause();
    push_dense_time += timeline.ElapsedSec();

    VLOG(3) << "push sparse and dense gradient done.";
    int32_t tmp_push_dense_wait_times = -1;
    int32_t tmp_push_sparse_wait_times = -1;
    static uint32_t push_dense_wait_times =
        static_cast<uint32_t>(tmp_push_dense_wait_times);
    static uint32_t push_sparse_wait_times =
        static_cast<uint32_t>(tmp_push_sparse_wait_times);
    if (push_dense_status_.size() >= push_dense_wait_times) {
      for (auto& t : push_dense_status_) {
        t.wait();
      }
      push_dense_status_.resize(0);
    }

    if (tmp_push_dense_wait_times == -1) {
      push_dense_status_.resize(0);
    }

    if (push_sparse_status_.size() >= push_sparse_wait_times) {
      for (auto& t : push_sparse_status_) {
        t.wait();
      }
      push_sparse_status_.resize(0);
    }

    if (tmp_push_sparse_wait_times == -1) {
      push_sparse_status_.resize(0);
    }
    VLOG(3) << "going to increase thread version";

    VLOG(3) << "push dense table id size: "
            << param_.program_config(0).push_dense_table_id_size();

    for (size_t i = 0; i < param_.program_config(0).push_dense_table_id_size();
         ++i) {
      uint64_t tid = static_cast<uint64_t>(
          param_.program_config(0).push_dense_table_id(i));
      pull_dense_worker_->IncreaseThreadVersion(thread_id_, tid);
    }

    thread_scope_->DropKids();
    ++batch_cnt;

    if (thread_id_ == 0) {
      // should be configured here
      if (batch_cnt > 0 && batch_cnt % 100 == 0) {
        for (size_t i = 0; i < op_total_time.size(); ++i) {
          fprintf(stderr, "op_name:[%zu][%s], op_mean_time:[%fs]\n", i,
                  op_name[i].c_str(), op_total_time[i] / batch_cnt);
        }
        fprintf(stderr, "mean read time: %fs\n", read_time / batch_cnt);
        fprintf(stderr, "IO percent: %f\n", read_time / total_time * 100);
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
  while ((cur_batch = device_reader_->Next()) > 0) {
    // pull sparse here
    for (size_t i = 0; i < param_.program_config(0).pull_sparse_table_id_size();
         ++i) {
      uint64_t tid = static_cast<uint64_t>(
          param_.program_config(0).pull_sparse_table_id(i));
      TableParameter table;
      for (auto i : param_.sparse_table()) {
        if (i.table_id() == tid) {
          table = i;
          break;
        }
      }
      fleet_ptr_->PullSparseVarsSync(*thread_scope_, tid,
                                     sparse_key_names_[tid], &features_[tid],
                                     &feature_values_[tid], table.fea_dim());
      CollectLabelInfo(i);
      FillSparseValue(i);
    }
    VLOG(3) << "fill sparse value for all sparse table done.";

    // do computation here
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

    // push gradients here
    for (size_t i = 0; i < param_.program_config(0).push_sparse_table_id_size();
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
          &feature_grads_[tid], &push_sparse_status_);
    }

    for (size_t i = 0; i < param_.program_config(0).push_dense_table_id_size();
         ++i) {
      uint64_t tid = static_cast<uint64_t>(
          param_.program_config(0).push_dense_table_id(i));
      fleet_ptr_->PushDenseVarsAsync(
          *thread_scope_, tid, dense_grad_names_[tid], &push_sparse_status_);
    }

    VLOG(3) << "push sparse and dense gradient done.";
    // the following code should be more precise and clean
    // TODO(guru4elephant)
    int32_t tmp_push_dense_wait_times = -1;
    int32_t tmp_push_sparse_wait_times = -1;
    static uint32_t push_dense_wait_times =
        static_cast<uint32_t>(tmp_push_dense_wait_times);
    static uint32_t push_sparse_wait_times =
        static_cast<uint32_t>(tmp_push_sparse_wait_times);

    if (push_dense_status_.size() >= push_dense_wait_times) {
      for (auto& t : push_dense_status_) {
        t.wait();
      }
      push_dense_status_.resize(0);
    }

    if (tmp_push_dense_wait_times == -1) {
      push_dense_status_.resize(0);
    }

    if (push_sparse_status_.size() >= push_sparse_wait_times) {
      for (auto& t : push_sparse_status_) {
        t.wait();
      }
      push_sparse_status_.resize(0);
    }

    if (tmp_push_sparse_wait_times == -1) {
      push_sparse_status_.resize(0);
    }

    for (size_t i = 0; i < param_.program_config(0).push_dense_table_id_size();
         ++i) {
      uint64_t tid = static_cast<uint64_t>(
          param_.program_config(0).push_dense_table_id(i));
      pull_dense_worker_->IncreaseThreadVersion(thread_id_, tid);
    }

    thread_scope_->DropKids();
    ++batch_cnt;
  }
}

}  // end namespace framework
}  // end namespace paddle
