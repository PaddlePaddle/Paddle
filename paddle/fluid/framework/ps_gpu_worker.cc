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
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/fleet/heter_wrapper.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/string/string_helper.h"

#if (defined PADDLE_WITH_CUDA || defined PADDLE_WITH_XPU) && \
    (defined PADDLE_WITH_PSLIB)
#include "paddle/fluid/platform/cuda_device_guard.h"

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

namespace paddle {
namespace framework {

void PSGPUWorker::Initialize(const TrainerDesc& desc) {
  param_ = desc.downpour_param();
  mpi_rank_ = desc.mpi_rank();
  trainer_desc_ = desc;
  /*
  for (int i = 0; i < trainer_desc_.xpu_recv_list_size(); ++i) {
    send_var_list_.push_back(trainer_desc_.xpu_recv_list(i));
  }
  */
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

  fleet_ptr_ = PSGPUWrapper::GetInstance();
  fetch_config_ = desc.fetch_config();
  use_cvm_ = desc.use_cvm();
  // for sparse value accessor, embedding only
  no_cvm_ = desc.no_cvm();
  scale_datanorm_ = desc.scale_datanorm();
  dump_slot_ = desc.dump_slot();
  dump_fields_.resize(desc.dump_fields_size());
  for (int i = 0; i < desc.dump_fields_size(); ++i) {
    dump_fields_[i] = desc.dump_fields(i);
  }
  adjust_ins_weight_config_ = desc.adjust_ins_weight_config();
  need_dump_param_ = false;
  dump_param_.resize(desc.dump_param_size());
  for (int i = 0; i < desc.dump_param_size(); ++i) {
    dump_param_[i] = desc.dump_param(i);
  }
  if (desc.dump_param_size() != 0) {
    need_dump_param_ = true;
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
    copy_sparse_tables_.push_back(std::make_pair(src_table, dest_table));
  }
  for (int i = 0; i < copy_table_config_.src_dense_tables_size(); ++i) {
    uint64_t src_table = copy_table_config_.src_dense_tables(i);
    uint64_t dest_table = copy_table_config_.dest_dense_tables(i);
    VLOG(3) << "copy_dense_tables_ push back " << src_table << "->"
            << dest_table;
    copy_dense_tables_.push_back(std::make_pair(src_table, dest_table));
  }
  for (auto& m : copy_table_config_.table_denpendency_map()) {
    if (sparse_key_names_.find(m.key()) != sparse_key_names_.end()) {
      // currently only support one dependency
      for (auto& value : m.values()) {
        table_dependency_[m.key()] = value;
      }
    }
  }
  // pull_queue_ = paddle::framework::MakeChannel<std::shared_ptr<HeterTask>>();
  // push_queue_ = paddle::framework::MakeChannel<std::shared_ptr<HeterTask>>();
}

void PSGPUWorker::SetChannelWriter(ChannelObject<std::string>* queue) {
  writer_.Reset(queue);
}

void PSGPUWorker::SetNeedDump(bool need_dump_field) {
  need_dump_field_ = need_dump_field;
}

void PSGPUWorker::DumpParam() {}

void PSGPUWorker::CollectLabelInfo(size_t table_idx) {
  if (no_cvm_) {
    return;
  }
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

    // skip slots which do not have embedding
    Variable* emb_var =
        thread_scope_->FindVar(sparse_value_names_[table_id][i]);
    if (emb_var == nullptr) {
      continue;
    }

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

void PSGPUWorker::FillSparseValue(size_t table_idx) {
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
    if (var_emb == nullptr) {
      continue;
    }
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
      if (use_cvm_ || no_cvm_) {
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
        if (is_nid &&
            static_cast<size_t>(index) == tensor->lod()[0][nid_ins_index]) {
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
        if (is_nid &&
            static_cast<size_t>(index) == tensor->lod()[0][nid_ins_index]) {
          nid_show_.push_back(fea_value[fea_idx][0]);
          ++nid_ins_index;
        }
        fea_idx++;
      }
    }
  }
}

void PSGPUWorker::AdjustInsWeight(std::shared_ptr<HeterTask> task) {
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
  for (size_t i = 0; i < len; ++i) {
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

void PSGPUWorker::TrainFiles() {
  return;
}

void PSGPUWorker::ResetStat() {
  total_time_ = 0;
  read_time_ = 0;
  pack_time_ = 0;
  pull_sparse_local_time_ = 0;
  op_all_time_ = 0;
  xpu_op_time_ = 0;
  xpu_wait_time_ = 0;
  cpu_op_time_ = 0;
  collect_label_time_ = 0;
  fill_sparse_time_ = 0;
  push_sparse_time_ = 0;
  gpu_2_cpu_time_ = 0;
  cpu_2_gpu_time_ = 0;
  total_inst_ = 0;
}

void PSGPUWorker::ProduceTasks() {
  return;
}

}  // end namespace framework
}  // end namespace paddle
#endif
