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
#include "paddle/fluid/string/string_helper.h"

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

namespace paddle {
namespace framework {

void DownpourWorker::Initialize(const TrainerDesc& desc) {
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
  scale_datanorm_ = desc.scale_datanorm();
  dump_slot_ = desc.dump_slot();
  dump_fields_.resize(desc.dump_fields_size());
  for (int i = 0; i < desc.dump_fields_size(); ++i) {
    dump_fields_[i] = desc.dump_fields(i);
  }
  adjust_ins_weight_config_ = desc.adjust_ins_weight_config();
}

void DownpourWorker::SetChannelWriter(ChannelObject<std::string>* queue) {
  writer_.Reset(queue);
}

void DownpourWorker::SetNeedDump(bool need_dump_field) {
  need_dump_field_ = need_dump_field;
}

template <typename T>
std::string PrintLodTensorType(LoDTensor* tensor, int64_t start, int64_t end) {
  auto count = tensor->numel();
  if (start < 0 || end > count) {
    VLOG(3) << "access violation";
    return "access violation";
  }
  std::ostringstream os;
  for (int64_t i = start; i < end; i++) {
    os << ":" << tensor->data<T>()[i];
  }
  return os.str();
}

std::string PrintLodTensorIntType(LoDTensor* tensor, int64_t start,
                                  int64_t end) {
  auto count = tensor->numel();
  if (start < 0 || end > count) {
    VLOG(3) << "access violation";
    return "access violation";
  }
  std::ostringstream os;
  for (int64_t i = start; i < end; i++) {
    os << ":" << static_cast<uint64_t>(tensor->data<int64_t>()[i]);
  }
  return os.str();
}

std::string PrintLodTensor(LoDTensor* tensor, int64_t start, int64_t end) {
  std::string out_val;
  if (tensor->type() == proto::VarType::FP32) {
    out_val = PrintLodTensorType<float>(tensor, start, end);
  } else if (tensor->type() == proto::VarType::INT64) {
    out_val = PrintLodTensorIntType(tensor, start, end);
  } else if (tensor->type() == proto::VarType::FP64) {
    out_val = PrintLodTensorType<double>(tensor, start, end);
  } else {
    out_val = "unsupported type";
  }
  return out_val;
}

std::pair<int64_t, int64_t> GetTensorBound(LoDTensor* tensor, int index) {
  auto& dims = tensor->dims();
  if (tensor->lod().size() != 0) {
    auto& lod = tensor->lod()[0];
    return {lod[index] * dims[1], lod[index + 1] * dims[1]};
  } else {
    return {index * dims[1], (index + 1) * dims[1]};
  }
}

bool CheckValidOutput(LoDTensor* tensor, int batch_size) {
  auto& dims = tensor->dims();
  if (dims.size() != 2) return false;
  if (tensor->lod().size() != 0) {
    auto& lod = tensor->lod()[0];
    if (lod.size() != batch_size + 1) {
      return false;
    }
  } else {
    if (dims[0] != batch_size) {
      return false;
    }
  }
  return true;
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

  size_t global_index = 0;
  for (size_t i = 0; i < sparse_key_names_[table_id].size(); ++i) {
    VLOG(3) << "sparse_key_names_[" << i
            << "]: " << sparse_key_names_[table_id][i];
    Variable* fea_var = thread_scope_->FindVar(sparse_key_names_[table_id][i]);
    if (fea_var == nullptr) {
      continue;
    }
    LoDTensor* tensor = fea_var->GetMutable<LoDTensor>();
    CHECK(tensor != nullptr) << "tensor of var "
                             << sparse_key_names_[table_id][i] << " is null";
    int64_t* ids = tensor->data<int64_t>();
    size_t fea_idx = 0;
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

  std::vector<float> init_value(table.fea_dim());
  for (size_t i = 0; i < sparse_key_names_[table_id].size(); ++i) {
    std::string slot_name = sparse_key_names_[table_id][i];
    std::string emb_slot_name = sparse_value_names_[table_id][i];
    Variable* var = thread_scope_->FindVar(slot_name);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    CHECK(tensor != nullptr) << "tensor of var " << slot_name << " is null";
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

    bool is_nid = (adjust_ins_weight_config_.need_adjust() &&
                   adjust_ins_weight_config_.nid_slot() == emb_slot_name);
    if (is_nid) {
      nid_show_.clear();
    }
    int nid_ins_index = 0;

    for (int index = 0; index < len; ++index) {
      if (use_cvm_) {
        if (ids[index] == 0u) {
          memcpy(ptr + table.emb_dim() * index, init_value.data(),
                 sizeof(float) * table.emb_dim());
          if (is_nid) {
            nid_show_.push_back(-1);
            ++nid_ins_index;
          }
          continue;
        }
        memcpy(ptr + table.emb_dim() * index, fea_value[fea_idx].data(),
               sizeof(float) * table.emb_dim());
        if (is_nid && index == tensor->lod()[0][nid_ins_index]) {
          nid_show_.push_back(fea_value[fea_idx][0]);
          ++nid_ins_index;
        }
        fea_idx++;
      } else {
        if (ids[index] == 0u) {
          memcpy(ptr + table.emb_dim() * index, init_value.data() + 2,
                 sizeof(float) * table.emb_dim());
          if (is_nid) {
            nid_show_.push_back(-1);
            ++nid_ins_index;
          }
          continue;
        }
        memcpy(ptr + table.emb_dim() * index, fea_value[fea_idx].data() + 2,
               sizeof(float) * table.emb_dim());
        if (is_nid && index == tensor->lod()[0][nid_ins_index]) {
          nid_show_.push_back(fea_value[fea_idx][0]);
          ++nid_ins_index;
        }
        fea_idx++;
      }
    }
  }
}

void DownpourWorker::AdjustInsWeight() {
#ifdef _LINUX
  // check var and tensor not null
  if (!adjust_ins_weight_config_.need_adjust()) {
    VLOG(0) << "need_adjust=false, skip adjust ins weight";
    return;
  }
  Variable* nid_var =
      thread_scope_->FindVar(adjust_ins_weight_config_.nid_slot());
  if (nid_var == nullptr) {
    VLOG(0) << "nid slot var " << adjust_ins_weight_config_.nid_slot()
            << " is nullptr, skip adjust ins weight";
    return;
  }
  LoDTensor* nid_tensor = nid_var->GetMutable<LoDTensor>();
  if (nid_tensor == nullptr) {
    VLOG(0) << "tensor of nid slot var " << adjust_ins_weight_config_.nid_slot()
            << " is nullptr, skip adjust ins weight";
    return;
  }
  Variable* ins_weight_var =
      thread_scope_->FindVar(adjust_ins_weight_config_.ins_weight_slot());
  if (ins_weight_var == nullptr) {
    VLOG(0) << "ins weight var " << adjust_ins_weight_config_.ins_weight_slot()
            << " is nullptr, skip adjust ins weight";
    return;
  }
  LoDTensor* ins_weight_tensor = ins_weight_var->GetMutable<LoDTensor>();
  if (ins_weight_tensor == nullptr) {
    VLOG(0) << "tensor of ins weight tensor "
            << adjust_ins_weight_config_.ins_weight_slot()
            << " is nullptr, skip adjust ins weight";
    return;
  }

  float* ins_weights = ins_weight_tensor->data<float>();
  size_t len = ins_weight_tensor->numel();  // len = batch size
  // here we assume nid_show slot only has one feasign in each instance
  CHECK(len == nid_show_.size()) << "ins_weight size should be equal to "
                                 << "nid_show size, " << len << " vs "
                                 << nid_show_.size();
  float nid_adjw_threshold = adjust_ins_weight_config_.nid_adjw_threshold();
  float nid_adjw_ratio = adjust_ins_weight_config_.nid_adjw_ratio();
  int64_t nid_adjw_num = 0;
  double nid_adjw_weight = 0.0;
  size_t ins_index = 0;
  for (int i = 0; i < len; ++i) {
    float nid_show = nid_show_[i];
    VLOG(3) << "nid_show " << nid_show;
    if (nid_show < 0) {
      VLOG(3) << "nid_show < 0, continue";
      continue;
    }
    float ins_weight = 1.0;
    if (nid_show >= 0 && nid_show < nid_adjw_threshold) {
      ins_weight = log(M_E +
                       (nid_adjw_threshold - nid_show) / nid_adjw_threshold *
                           nid_adjw_ratio);
      // count nid adjw insnum and weight
      ++nid_adjw_num;
      nid_adjw_weight += ins_weight;
      // choose large ins weight
      VLOG(3) << "ins weight new " << ins_weight << ", ins weight origin "
              << ins_weights[ins_index];
      if (ins_weight > ins_weights[ins_index]) {
        VLOG(3) << "ins " << ins_index << " weight changes to " << ins_weight;
        ins_weights[ins_index] = ins_weight;
      }
      ++ins_index;
    }
  }
  VLOG(3) << "nid adjw info: total_adjw_num: " << nid_adjw_num
          << ", avg_adjw_weight: " << nid_adjw_weight;
#endif
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
  double adjust_ins_weight_time = 0.0;
  double collect_label_time = 0.0;
  double fill_sparse_time = 0.0;
  double push_sparse_time = 0.0;
  double push_dense_time = 0.0;
  int cur_batch;
  int batch_cnt = 0;
  uint64_t total_inst = 0;
  timeline.Start();
  while ((cur_batch = device_reader_->Next()) > 0) {
    timeline.Pause();
    read_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();
    VLOG(3) << "program config size: " << param_.program_config_size();
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
      timeline.Start();
      fleet_ptr_->PullSparseVarsSync(*thread_scope_, tid,
                                     sparse_key_names_[tid], &features_[tid],
                                     &feature_values_[tid], table.fea_dim());
      timeline.Pause();
      pull_sparse_time += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
      timeline.Start();
      CollectLabelInfo(i);
      timeline.Pause();
      collect_label_time += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
      timeline.Start();
      FillSparseValue(i);
      timeline.Pause();
      fill_sparse_time += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
      timeline.Start();
      auto nid_iter = std::find(sparse_value_names_[tid].begin(),
                                sparse_value_names_[tid].end(),
                                adjust_ins_weight_config_.nid_slot());
      if (nid_iter != sparse_value_names_[tid].end()) {
        AdjustInsWeight();
      }
      timeline.Pause();
      adjust_ins_weight_time += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
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
        VLOG(3) << "Going to run op " << op_name[run_op_idx];
        op->Run(*thread_scope_, place_);
        VLOG(3) << "Op " << op_name[run_op_idx] << " Finished";
        timeline.Pause();
        op_total_time[run_op_idx++] += timeline.ElapsedSec();
        total_time += timeline.ElapsedSec();
      }
    }

    if (need_to_push_sparse_) {
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
        timeline.Start();
        fleet_ptr_->PushSparseVarsWithLabelAsync(
            *thread_scope_, tid, features_[tid], feature_labels_[tid],
            sparse_key_names_[tid], sparse_grad_names_[tid], table.emb_dim(),
            &feature_grads_[tid], &push_sparse_status_, cur_batch, use_cvm_,
            dump_slot_);
        timeline.Pause();
        push_sparse_time += timeline.ElapsedSec();
        total_time += timeline.ElapsedSec();
      }
    }

    if (need_to_push_dense_) {
      timeline.Start();
      for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
           ++i) {
        uint64_t tid = static_cast<uint64_t>(
            param_.program_config(0).push_dense_table_id(i));
        fleet_ptr_->PushDenseVarsAsync(
            *thread_scope_, tid, dense_grad_names_[tid], &push_sparse_status_,
            scale_datanorm_, cur_batch);
      }
      timeline.Pause();
      push_dense_time += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
      VLOG(3) << "push sparse and dense gradient done.";
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

      VLOG(3) << "going to increase thread version";
      VLOG(3) << "push dense table id size: "
              << param_.program_config(0).push_dense_table_id_size();
    }

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
          fprintf(stderr, "op_name:[%zu][%s], op_mean_time:[%fs]\n", i,
                  op_name[i].c_str(), op_total_time[i] / batch_cnt);
          if (op_to_time.find(op_name[i]) == op_to_time.end()) {
            op_to_time[op_name[i]] = 0.0;
          }
          op_to_time[op_name[i]] += op_total_time[i];
          op_sum_time += op_total_time[i];
        }
        for (auto& i : op_to_time) {
          fprintf(stderr, "op [%s] run total time: [%f]ms\n", i.first.c_str(),
                  i.second / batch_cnt);
        }
        fprintf(stderr, "op run total time: %fs\n", op_sum_time / batch_cnt);
        fprintf(stderr, "train total time: %fs\n", total_time / batch_cnt);
        fprintf(stderr, "pull sparse time: %fs\n",
                pull_sparse_time / batch_cnt);
        fprintf(stderr, "fill sparse time: %fs\n",
                fill_sparse_time / batch_cnt);
        fprintf(stderr, "push sparse time: %fs\n",
                push_sparse_time / batch_cnt);
        fprintf(stderr, "push dense time: %fs\n", push_dense_time / batch_cnt);
        fprintf(stderr, "collect label time: %fs\n",
                collect_label_time / batch_cnt);
        fprintf(stderr, "adjust ins weight time: %fs\n",
                adjust_ins_weight_time / batch_cnt);
        fprintf(stderr, "mean read time: %fs\n", read_time / batch_cnt);
        fprintf(stderr, "IO percent: %f\n", read_time / total_time * 100);
        fprintf(stderr, "op run percent: %f\n", op_sum_time / total_time * 100);
        fprintf(stderr, "pull sparse time percent: %f\n",
                pull_sparse_time / total_time * 100);
        fprintf(stderr, "adjust ins weight time percent: %f\n",
                adjust_ins_weight_time / total_time * 100);
        fprintf(stderr, "collect label time percent: %f\n",
                collect_label_time / total_time * 100);
        fprintf(stderr, "fill sparse time percent: %f\n",
                fill_sparse_time / total_time * 100);
        fprintf(stderr, "push sparse time percent: %f\n",
                push_sparse_time / total_time * 100);
        fprintf(stderr, "push dense time percent: %f\n",
                push_dense_time / total_time * 100);
        fprintf(stderr, "%6.2f instances/s\n", total_inst / total_time);
      }
    }
    timeline.Start();
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
      fleet_ptr_->PullSparseVarsSync(*thread_scope_, tid,
                                     sparse_key_names_[tid], &features_[tid],
                                     &feature_values_[tid], table.fea_dim());
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
            dump_slot_);
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
      int batch_size = device_reader_->GetCurBatchSize();
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
        for (int i = 0; i < batch_size; ++i) {
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
    }

    PrintFetchVars();
    thread_scope_->DropKids();
    ++batch_cnt;
  }
  if (need_dump_field_) {
    writer_.Flush();
  }
}

}  // end namespace framework
}  // end namespace paddle
