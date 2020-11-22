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

void HeterBoxWorker::Initialize(const TrainerDesc& desc) {
  param_ = desc.downpour_param();
  mpi_rank_ = desc.mpi_rank();
  trainer_desc_ = desc;
  for (int i = 0; i < trainer_desc_.xpu_recv_list_size(); ++i) {
    send_var_list_.push_back(trainer_desc_.xpu_recv_list(i));
  }
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
  pull_queue_ = paddle::framework::MakeChannel<std::shared_ptr<HeterTask>>();
  push_queue_ = paddle::framework::MakeChannel<std::shared_ptr<HeterTask>>();
}

void HeterBoxWorker::SetChannelWriter(ChannelObject<std::string>* queue) {
  writer_.Reset(queue);
}

void HeterBoxWorker::SetNeedDump(bool need_dump_field) {
  need_dump_field_ = need_dump_field;
}

void HeterBoxWorker::DumpParam() {}

void HeterBoxWorker::CollectLabelInfo(std::shared_ptr<HeterTask> task,
                                      size_t table_idx) {
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
  auto& feature = (task->features_)[table_id];
  auto& feature_label = (task->feature_labels_)[table_id];
  Scope* scope = task->scope_;
  feature_label.resize(feature.size());
  Variable* var = scope->FindVar(label_var_name_[table_id]);
  LoDTensor* tensor = var->GetMutable<LoDTensor>();
  int64_t* label_ptr = tensor->data<int64_t>();

  size_t global_index = 0;
  for (size_t i = 0; i < sparse_key_names_[table_id].size(); ++i) {
    VLOG(3) << "sparse_key_names_[" << i
            << "]: " << sparse_key_names_[table_id][i];
    Variable* fea_var = scope->FindVar(sparse_key_names_[table_id][i]);
    if (fea_var == nullptr) {
      continue;
    }
    LoDTensor* tensor = fea_var->GetMutable<LoDTensor>();
    CHECK(tensor != nullptr) << "tensor of var "
                             << sparse_key_names_[table_id][i] << " is null";

    // skip slots which do not have embedding
    Variable* emb_var = scope->FindVar(sparse_value_names_[table_id][i]);
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

void HeterBoxWorker::FillSparseValue(std::shared_ptr<HeterTask> task,
                                     size_t table_idx) {
  uint64_t table_id = static_cast<uint64_t>(
      param_.program_config(0).pull_sparse_table_id(table_idx));

  TableParameter table;
  for (auto i : param_.sparse_table()) {
    if (i.table_id() == table_id) {
      table = i;
      break;
    }
  }

  auto& fea_value = (task->feature_values_)[table_id];
  Scope* scope = task->scope_;
  auto fea_idx = 0u;

  std::vector<float> init_value(table.fea_dim());
  for (size_t i = 0; i < sparse_key_names_[table_id].size(); ++i) {
    std::string slot_name = sparse_key_names_[table_id][i];
    std::string emb_slot_name = sparse_value_names_[table_id][i];
    Variable* var = scope->FindVar(slot_name);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    CHECK(tensor != nullptr) << "tensor of var " << slot_name << " is null";
    int64_t* ids = tensor->data<int64_t>();
    int len = tensor->numel();
    Variable* var_emb = scope->FindVar(emb_slot_name);
    if (var_emb == nullptr) {
      continue;
    }
    LoDTensor* tensor_emb = var_emb->GetMutable<LoDTensor>();
    float* ptr = tensor_emb->mutable_data<float>({len, table.emb_dim()},
                                                 platform::CPUPlace());
    // memset(ptr, 0, sizeof(float) * len * table.emb_dim());
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

void HeterBoxWorker::AdjustInsWeight(std::shared_ptr<HeterTask> task) {
#ifdef _LINUX
  // check var and tensor not null
  Scope* scope = task->scope_;
  if (!adjust_ins_weight_config_.need_adjust()) {
    VLOG(0) << "need_adjust=false, skip adjust ins weight";
    return;
  }
  Variable* nid_var = scope->FindVar(adjust_ins_weight_config_.nid_slot());
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
      scope->FindVar(adjust_ins_weight_config_.ins_weight_slot());
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

void HeterBoxWorker::TrainFiles() {
  VLOG(3) << "Begin to train files";
  platform::SetNumThreads(1);
  need_to_push_dense_ = false;
  while (1) {
    VLOG(3) << "before heter task";
    std::shared_ptr<HeterTask> task;

    if (!pull_queue_->Get(task)) {
      VLOG(3) << "get task";
      break;
    }
    VLOG(3) << "get task done";
    Scope* scope = task->scope_->kids().front();
    VLOG(3) << "get kid done";
    // do computation here
    task->timeline.Start();
    for (auto& op : ops_) {
      if (op->HasAttr("op_device")) {
        auto device = op->Attr<std::string>("op_device");
        if (device != "gpu") {
          continue;
        }
      }
      bool need_skip = false;
      for (auto t = 0u; t < skip_ops_.size(); ++t) {
        if (op->Type().find(skip_ops_[t]) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      if (!need_skip) {
        op->Run(*(scope), place_);
      }
    }
    platform::DeviceContextPool::Instance().Get(place_)->Wait();
    task->timeline.Pause();
    task->xpu_op_time += task->timeline.ElapsedSec();
    task->total_time += task->timeline.ElapsedSec();
    push_queue_->Put(task);
  }
}

void HeterTask::PackGpuTask(Scope* thread_scope, DataFeed* reader,
                            const ProgramDesc& program) {
  auto& block = program.Block(0);
  if (!scope_) {
    scope_ = &(thread_scope->NewScope());
    for (auto& var : block.AllVars()) {
      if (!var->Persistable()) {
        auto* ptr = scope_->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
      }
    }
  }
  reader->AssignFeedVar(*scope_);
  cur_batch_ = reader->Next();
}

void HeterBoxWorker::ResetStat() {
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

void HeterBoxWorker::ProduceTasks() {
  need_to_push_dense_ = false;
  while (1) {
    std::shared_ptr<HeterTask> task;
    task = object_pool_.Get();
    task->Reset();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      task->timeline.Start();
      task->PackGpuTask(thread_scope_, device_reader_, program_);
      task->timeline.Pause();
      task->pack_time = task->timeline.ElapsedSec();
      task->total_time += task->pack_time;
      if (task->cur_batch_ <= 0) {
        if (!pull_queue_->Closed() && batch_cnt_ == done_cnt_) {
          pull_queue_->Close();
        }
        break;
      }
      batch_cnt_ += 1;
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
      task->timeline.Start();
      fleet_ptr_->HeterPullSparseVars(thread_id_, task, tid,
                                      sparse_key_names_[tid], table.fea_dim(),
                                      sparse_value_names_[tid]);
      task->timeline.Pause();
      task->pull_sparse_local_time += task->timeline.ElapsedSec();
      task->total_time += task->timeline.ElapsedSec();

      task->timeline.Start();
      CollectLabelInfo(task, i);
      task->timeline.Pause();
      task->collect_label_time += task->timeline.ElapsedSec();
      task->total_time += task->timeline.ElapsedSec();

      task->timeline.Start();
      FillSparseValue(task, i);
      task->timeline.Pause();
      task->fill_sparse_time += task->timeline.ElapsedSec();
      task->total_time += task->timeline.ElapsedSec();

      auto nid_iter = std::find(sparse_value_names_[tid].begin(),
                                sparse_value_names_[tid].end(),
                                adjust_ins_weight_config_.nid_slot());
      if (nid_iter != sparse_value_names_[tid].end()) {
        AdjustInsWeight(task);
      }
    }

    task->timeline.Start();
    size_t op_index = 0;
    for (; op_index < ops_.size(); ++op_index) {
      auto& op = ops_[op_index];
      if (op->HasAttr("op_device")) {
        auto device = op->Attr<std::string>("op_device");
        if (device == "gpu") {
          break;
        }
      }
      bool need_skip = false;
      for (auto t = 0u; t < skip_ops_.size(); ++t) {
        if (op->Type().find(skip_ops_[t]) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      if (!need_skip) {
        op->Run(*(task->scope_), platform::CPUPlace());
      }
    }

    task->timeline.Pause();
    task->cpu_op_time += task->timeline.ElapsedSec();
    task->total_time += task->timeline.ElapsedSec();

    task->timeline.Start();
    // prepare for gpu
    Scope* cpu_scope = task->scope_;
    Scope* gpu_scope = nullptr;
    if (cpu_scope->kids().empty()) {
      gpu_scope = &cpu_scope->NewScope();
    } else {
      gpu_scope = cpu_scope->kids().front();
    }
    for (const std::string& name : send_var_list_) {
      const LoDTensor& cpu_tensor = cpu_scope->FindVar(name)->Get<LoDTensor>();
      LoDTensor* gpu_tensor = gpu_scope->Var(name)->GetMutable<LoDTensor>();
      gpu_tensor->set_lod(cpu_tensor.lod());
      gpu_tensor->Resize(cpu_tensor.dims());
      gpu_tensor->set_layout(cpu_tensor.layout());
      void* gpu_ptr = gpu_tensor->mutable_data(place_, cpu_tensor.type());
      const void* cpu_ptr = cpu_tensor.data<void>();
      memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, place_), gpu_ptr,
                   platform::CPUPlace(), cpu_ptr,
                   cpu_tensor.numel() * SizeOfType(cpu_tensor.type()),
                   copy_stream_);
    }
    task->timeline.Pause();
    task->cpu_2_gpu_time += task->timeline.ElapsedSec();
    task->total_time += task->timeline.ElapsedSec();
    pull_queue_->Put(task);
    push_queue_->Get(task);

    int need_copy_grad = 1;
    task->timeline.Start();
    for (; op_index < ops_.size(); ++op_index) {
      auto& op = ops_[op_index];
      if (op->HasAttr("op_device")) {
        auto device = op->Attr<std::string>("op_device");
        if (device == "gpu") {
          continue;
        }
      }
      bool need_skip = false;
      for (auto t = 0u; t < skip_ops_.size(); ++t) {
        if (op->Type().find(skip_ops_[t]) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      if (!need_skip) {
        need_copy_grad = 0;
        op->Run(*(task->scope_), platform::CPUPlace());
      }
    }
    task->timeline.Pause();
    task->cpu_op_time += task->timeline.ElapsedSec();
    task->total_time += task->timeline.ElapsedSec();

    VLOG(3) << "fill sparse value for all sparse table done.";
    for (std::string& var_name : check_nan_var_names_) {
      Variable* var = (task->scope_)->FindVar(var_name);
      if (var == nullptr) {
        continue;
      }
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      if (tensor == nullptr) {
        continue;
      }
      PADDLE_ENFORCE_EQ(framework::TensorContainsInf(*tensor), false,
                        platform::errors::InvalidArgument(
                            "Tensor %s contains Inf.", var_name));
      PADDLE_ENFORCE_EQ(framework::TensorContainsNAN(*tensor), false,
                        platform::errors::InvalidArgument(
                            "Tensor %s contains NAN.", var_name));
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
        Scope* src_scope = task->scope_;
        Scope* dest_scope = nullptr;
        task->timeline.Start();
        if (need_copy_grad) {
          if (cpu_scope->kids().empty()) {
            dest_scope = &src_scope->NewScope();
          } else {
            dest_scope = src_scope->kids().front();
          }
          auto dev_id = BOOST_GET_CONST(platform::CUDAPlace, place_).device;
          platform::CUDADeviceGuard guard(dev_id);

          for (const std::string& name : sparse_grad_names_[tid]) {
            const LoDTensor& src_tensor =
                src_scope->FindVar(name)->Get<LoDTensor>();
            LoDTensor* dest_tensor =
                dest_scope->Var(name)->GetMutable<LoDTensor>();
            dest_tensor->set_lod(src_tensor.lod());
            dest_tensor->Resize(src_tensor.dims());
            dest_tensor->set_layout(src_tensor.layout());
            void* dest_ptr = dest_tensor->mutable_data(platform::CPUPlace(),
                                                       src_tensor.type());
            const void* src_ptr = src_tensor.data<void>();
            memory::Copy(platform::CPUPlace(), dest_ptr,
                         BOOST_GET_CONST(platform::CUDAPlace, place_), src_ptr,
                         src_tensor.numel() * SizeOfType(src_tensor.type()),
                         copy_stream_);
          }
        } else {
          dest_scope = task->scope_;
        }
        task->timeline.Pause();
        task->gpu_2_cpu_time += task->timeline.ElapsedSec();
        task->total_time += task->timeline.ElapsedSec();

        task->timeline.Start();
        fleet_ptr_->HeterPushSparseVars(
            task, *(dest_scope), tid, sparse_key_names_[tid],
            sparse_grad_names_[tid], table.emb_dim(), &push_sparse_status_,
            use_cvm_, dump_slot_, no_cvm_);
        task->timeline.Pause();
        task->push_sparse_time += task->timeline.ElapsedSec();
        task->total_time += task->timeline.ElapsedSec();
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
    {
      std::lock_guard<std::mutex> lock(mutex_);
      total_time_ += task->total_time;
      read_time_ += task->read_time;
      pack_time_ += task->pack_time;
      pull_sparse_local_time_ += task->pull_sparse_local_time;
      op_all_time_ += task->op_all_time;
      xpu_op_time_ += task->xpu_op_time;
      xpu_wait_time_ += task->xpu_wait_time;
      cpu_op_time_ += task->cpu_op_time;
      collect_label_time_ += task->collect_label_time;
      fill_sparse_time_ += task->fill_sparse_time;
      push_sparse_time_ += task->push_sparse_time;
      gpu_2_cpu_time_ += task->gpu_2_cpu_time;
      cpu_2_gpu_time_ += task->cpu_2_gpu_time;
      total_inst_ += task->cur_batch_;
    }
    done_cnt_.fetch_add(1, std::memory_order_relaxed);
    if (thread_id_ == 0) {
      // should be configured here
      if (done_cnt_ > 0 && done_cnt_ % 100 == 0) {
        fprintf(stderr, "cpu_2_gpu total time: %fs\n",
                cpu_2_gpu_time_ / done_cnt_);
        fprintf(stderr, "gpu_2_cpu run total time: %fs\n",
                gpu_2_cpu_time_ / done_cnt_);
        fprintf(stderr, "cpu op run total time: %fs\n",
                cpu_op_time_ / done_cnt_);
        fprintf(stderr, "xpu op run total time: %fs\n",
                xpu_op_time_ / done_cnt_);
        fprintf(stderr, "xpu wait total time: %fs\n",
                xpu_wait_time_ / done_cnt_);
        fprintf(stderr, "pack task time: %fs\n", pack_time_ / done_cnt_);
        fprintf(stderr, "train total time: %fs\n", total_time_ / done_cnt_);
        fprintf(stderr, "pull sparse local time: %fs\n",
                pull_sparse_local_time_ / done_cnt_);
        fprintf(stderr, "fill sparse time: %fs\n",
                fill_sparse_time_ / done_cnt_);
        fprintf(stderr, "push sparse time: %fs\n",
                push_sparse_time_ / done_cnt_);
        fprintf(stderr, "collect label time: %fs\n",
                collect_label_time_ / done_cnt_);
        fprintf(stderr, "mean read time: %fs\n", read_time_ / done_cnt_);
        fprintf(stderr, "IO percent: %f\n", read_time_ / total_time_ * 100);
        fprintf(stderr, "cpu_2_gpu run percent: %f\n",
                cpu_2_gpu_time_ / total_time_ * 100);
        fprintf(stderr, "gpu_2_cpu run percent: %f\n",
                gpu_2_cpu_time_ / total_time_ * 100);
        fprintf(stderr, "cpu op run percent: %f\n",
                cpu_op_time_ / total_time_ * 100);
        fprintf(stderr, "xpu op run percent: %f\n",
                xpu_op_time_ / total_time_ * 100);
        fprintf(stderr, "xpu wait percent: %f\n",
                xpu_wait_time_ / total_time_ * 100);
        fprintf(stderr, "pack task percent: %f\n",
                pack_time_ / total_time_ * 100);
        fprintf(stderr, "pull sparse local time percent: %f\n",
                pull_sparse_local_time_ / total_time_ * 100);
        fprintf(stderr, "collect label time percent: %f\n",
                collect_label_time_ / total_time_ * 100);
        fprintf(stderr, "fill sparse time percent: %f\n",
                fill_sparse_time_ / total_time_ * 100);
        fprintf(stderr, "push sparse time percent: %f\n",
                push_sparse_time_ / total_time_ * 100);
        fprintf(stderr, "%6.2f instances/s\n", total_inst_ / total_time_);
      }
    }

    VLOG(3) << "done taskid = " << task->taskid_;
    task->scope_->DropKids();
    object_pool_.Push(task);
  }
}

}  // end namespace framework
}  // end namespace paddle
#endif
