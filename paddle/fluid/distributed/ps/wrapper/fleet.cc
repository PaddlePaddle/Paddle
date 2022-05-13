/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <google/protobuf/text_format.h>

#include "paddle/fluid/distributed/ps/service/communicator/communicator.h"
#include "paddle/fluid/distributed/ps/table/table.h"
#include "paddle/fluid/distributed/ps/wrapper/fleet.h"

namespace paddle {
namespace distributed {

using framework::LoDTensor;
using framework::ProgramDesc;
using framework::VarDesc;
using framework::Variable;

const uint32_t MAX_FEASIGN_NUM = 1024 * 100 * 100;
std::shared_ptr<FleetWrapper> FleetWrapper::s_instance_ = NULL;
bool FleetWrapper::is_initialized_ = false;

std::shared_ptr<paddle::distributed::PSCore> FleetWrapper::pserver_ptr_ = NULL;
std::shared_ptr<paddle::distributed::PSClient> FleetWrapper::worker_ptr_ = NULL;

int FleetWrapper::RegisterHeterCallback(HeterCallBackFunc handler) {
  VLOG(0) << "RegisterHeterCallback support later";
  return 0;
}

int32_t FleetWrapper::CopyTable(const uint64_t src_table_id,
                                const uint64_t dest_table_id) {
  VLOG(0) << "CopyTable support later";
  return 0;
}

int32_t FleetWrapper::CopyTableByFeasign(
    const uint64_t src_table_id, const uint64_t dest_table_id,
    const std::vector<uint64_t>& feasign_list) {
  VLOG(0) << "CopyTableByFeasign support later";
  return 0;
}

void FleetWrapper::SetClient2ClientConfig(int request_timeout_ms,
                                          int connect_timeout_ms,
                                          int max_retry) {
  client2client_request_timeout_ms_ = request_timeout_ms;
  client2client_connect_timeout_ms_ = connect_timeout_ms;
  client2client_max_retry_ = max_retry;
}

void FleetWrapper::LoadSparseOnServer(const std::string& path,
                                      const std::string& meta,
                                      uint32_t table_id) {
  VLOG(3) << "load sparse table " << table_id << " with " << path << " meta "
          << meta;
  pserver_ptr_->_server_ptr->GetTable(table_id)->Load(path, meta);
}

void FleetWrapper::InitServer(
    const std::string& dist_desc,
    const std::vector<std::string>& host_sign_list, int index, int trainers,
    const std::vector<framework::ProgramDesc>& server_sub_program) {
  if (!is_initialized_) {
    VLOG(3) << "Going to init server";
    pserver_ptr_ = std::shared_ptr<paddle::distributed::PSCore>(
        new paddle::distributed::PSCore());
    pserver_ptr_->InitServer(dist_desc, &host_sign_list, host_sign_list.size(),
                             index, trainers, server_sub_program);
    is_initialized_ = true;
  } else {
    VLOG(3) << "Server can be initialized only once";
  }
}

void FleetWrapper::InitGFlag(const std::string& gflags) {
  VLOG(3) << "Init With Gflags:" << gflags;
  std::vector<std::string> flags = paddle::string::split_string(gflags);
  if (flags.size() < 1) {
    flags.push_back("-max_body_size=314217728");
    flags.push_back("-bthread_concurrency=40");
    flags.push_back("-socket_max_unwritten_bytes=2048000000");
    flags.push_back("-max_connection_pool_size=1950");
  }
  auto it = flags.begin();
  flags.insert(it, "exe default");
  char* flags_ptr[flags.size()];
  for (size_t i = 0; i < flags.size(); ++i) {
    flags_ptr[i] = (char*)(flags[i].c_str());  // NOLINT
  }
  int params_cnt = flags.size();
  char** params_ptr = &(flags_ptr[0]);
  ::GFLAGS_NAMESPACE::ParseCommandLineFlags(&params_cnt, &params_ptr, true);
}

void FleetWrapper::InitWorker(const std::string& dist_desc,
                              const std::vector<std::string>& host_sign_list,
                              int index) {
  if (!is_initialized_) {
    // not used, just for psclient's init
    // TODO(zhaocaibei123): remove this later
    std::map<uint64_t, std::vector<paddle::distributed::Region>>
        dense_pull_regions;

    if (worker_ptr_.get() == nullptr) {
      paddle::distributed::PSParameter ps_param;
      google::protobuf::TextFormat::ParseFromString(dist_desc, &ps_param);
      InitGFlag(ps_param.init_gflags());
      int servers = host_sign_list.size();
      ps_env_.SetPsServers(&host_sign_list, servers);
      worker_ptr_ = std::shared_ptr<paddle::distributed::PSClient>(
          paddle::distributed::PSClientFactory::Create(ps_param));
      worker_ptr_->Configure(ps_param, dense_pull_regions, ps_env_, index);
    }
  } else {
    VLOG(3) << "Client can be initialized only once";
  }
}

void FleetWrapper::StopServer() {
  VLOG(3) << "Going to stop server";
  auto status = worker_ptr_->StopServer();
  status.wait();
}

void FleetWrapper::FinalizeWorker() {
  VLOG(3) << "Going to finalize worker";
  worker_ptr_->FinalizeWorker();
}

void FleetWrapper::BarrierWithTable(uint32_t barrier_type) {
  VLOG(3) << "Going to Barrier worker";
  auto* communicator = Communicator::GetInstance();
  communicator->BarrierWithTable(barrier_type);
}

uint64_t FleetWrapper::RunServer(const std::string& ip, uint32_t port) {
  VLOG(3) << "Going to run server with ip " << ip << " port " << port;
  auto ret = pserver_ptr_->RunServer(ip, port);
  return ret;
}

std::vector<uint64_t> FleetWrapper::GetClientsInfo() {
  VLOG(3) << "Going to get client info";
  std::vector<uint64_t> res = ps_env_.GetClientInfo();
  for (auto rr : res) {
    VLOG(2) << "FleetWrapper::GetClientInfo " << rr;
  }
  return res;
}

int FleetWrapper::SetClients(std::vector<uint64_t>& host_sign_list) {
  int node = host_sign_list.size();
  return ps_env_.SetPsClients(host_sign_list.data(), node);
}

void FleetWrapper::CreateClient2ClientConnection() {
  VLOG(1) << "Going to create client2client connection";
  worker_ptr_->CreateClient2ClientConnection(client2client_request_timeout_ms_,
                                             client2client_connect_timeout_ms_,
                                             client2client_max_retry_);
}

std::future<int32_t> FleetWrapper::PullSparseVarsAsync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<std::string>& var_names, std::vector<uint64_t>* fea_keys,
    std::vector<std::vector<float>>* fea_values, int fea_value_dim) {
  fea_keys->clear();
  fea_keys->resize(0);
  fea_keys->reserve(MAX_FEASIGN_NUM);
  for (auto name : var_names) {
    Variable* var = scope.FindVar(name);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    CHECK(tensor != nullptr) << "tensor of var " << name << " is null";
    int64_t* ids = tensor->data<int64_t>();
    size_t len = tensor->numel();
    for (auto i = 0u; i < len; ++i) {
      if (ids[i] == 0u) {
        continue;
      }
      fea_keys->push_back(static_cast<uint64_t>(ids[i]));
    }
  }
  fea_values->resize(fea_keys->size() + 1);
  for (auto& t : *fea_values) {
    t.resize(fea_value_dim);
  }
  std::vector<float*> pull_result_ptr;
  for (auto& t : *fea_values) {
    pull_result_ptr.push_back(t.data());
  }

  bool training = true;
  return pserver_ptr_->_worker_ptr->PullSparse(pull_result_ptr.data(), table_id,
                                               fea_keys->data(),
                                               fea_keys->size(), training);
}

void FleetWrapper::PullSparseVarsSync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<std::string>& var_names, std::vector<uint64_t>* fea_keys,
    std::vector<std::vector<float>>* fea_values, int fea_value_dim,
    const std::vector<std::string>& var_emb_names) {
  std::vector<std::future<int32_t>> pull_sparse_status;
  pull_sparse_status.resize(0);
  fea_keys->clear();
  fea_keys->resize(0);
  fea_keys->reserve(MAX_FEASIGN_NUM);
  for (size_t var_index = 0; var_index < var_names.size(); ++var_index) {
    const std::string& name = var_names[var_index];
    Variable* var = scope.FindVar(name);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    CHECK(tensor != nullptr) << "tensor of var " << name << " is null";
    int64_t* ids = tensor->data<int64_t>();
    size_t len = tensor->numel();

    // skip slots which do not have embedding
    const std::string& emb_name = var_emb_names[var_index];
    Variable* emb_var = scope.FindVar(emb_name);
    if (emb_var == nullptr) {
      continue;
    }

    for (auto i = 0u; i < len; ++i) {
      if (ids[i] == 0u) {
        continue;
      }
      fea_keys->push_back(static_cast<uint64_t>(ids[i]));
    }
  }
  fea_values->resize(fea_keys->size() + 1);
  for (auto& t : *fea_values) {
    t.resize(fea_value_dim);
  }
  std::vector<float*> pull_result_ptr;
  for (auto& t : *fea_values) {
    pull_result_ptr.push_back(t.data());
  }
  bool training = true;
  auto status = pserver_ptr_->_worker_ptr->PullSparse(
      pull_result_ptr.data(), table_id, fea_keys->data(), fea_keys->size(),
      training);
  pull_sparse_status.push_back(std::move(status));
  for (auto& t : pull_sparse_status) {
    t.wait();
    auto status = t.get();
    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      sleep(sleep_seconds_before_fail_exit_);
      exit(-1);
    }
  }
}

// is_training is true means training, false means inference, the behavior is
// different on pserver

void FleetWrapper::PullSparseToTensorSync(const uint64_t table_id, int fea_dim,
                                          uint64_t padding_id,
                                          platform::Place place,
                                          bool is_training,
                                          std::vector<const LoDTensor*>* inputs,
                                          std::vector<LoDTensor*>* outputs) {
  std::vector<uint64_t> fea_keys;
  std::vector<float*> pull_result_ptr;
  fea_keys.reserve(MAX_FEASIGN_NUM / 100);
  pull_result_ptr.reserve(MAX_FEASIGN_NUM / 100);
  std::vector<float> init_value(fea_dim, 0);
  framework::LoDTensor* output = nullptr;
  float* output_data = nullptr;
  size_t output_index = -1;
  size_t output_len = 0;
  for (size_t index = 0; index < inputs->size(); ++index) {
    const framework::LoDTensor* tensor = inputs->at(index);
    const int64_t* ids = tensor->data<int64_t>();
    size_t len = tensor->numel();
    for (size_t i = 0; i < len; ++i, output_len += fea_dim) {
      if (!output || output_len == size_t(output->numel())) {
        ++output_index;
        CHECK(output_index < outputs->size());  // NOLINT
        output = outputs->at(output_index);
        output->set_lod(tensor->lod());
        output_data = output->mutable_data<float>(place);
        output_len = 0;
        CHECK(output->numel() % fea_dim == 0);  // NOLINT
        CHECK(output_data != nullptr);          // NOLINT
      }
      uint64_t real_id = static_cast<uint64_t>(ids[i]);
      if (real_id == padding_id) {
        memcpy(output_data + output_len, init_value.data(),
               sizeof(float) * fea_dim);
        continue;
      }
      fea_keys.push_back(real_id);
      pull_result_ptr.push_back(output_data + output_len);
    }
  }

  auto status =
      worker_ptr_->PullSparse(pull_result_ptr.data(), table_id, fea_keys.data(),
                              fea_keys.size(), is_training);
  status.wait();
  auto ret = status.get();
  if (ret != 0) {
    LOG(ERROR) << "fleet pull sparse failed, status[" << ret << "]";
    sleep(sleep_seconds_before_fail_exit_);
  }
}

void FleetWrapper::PullDenseVarsAsync(
    const Scope& scope, const uint64_t tid,
    const std::vector<std::string>& var_names,
    std::vector<std::future<int32_t>>* pull_dense_status, bool in_cpu) {
  auto& regions = regions_[tid];
  regions.clear();
  regions.resize(var_names.size());
  for (auto i = 0u; i < var_names.size(); ++i) {
    std::string varname = var_names[i];
    if (!in_cpu) {
      varname = var_names[i] + "pin";
    }
    Variable* var = scope.FindVar(varname);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* w = tensor->data<float>();
    paddle::distributed::Region reg(w, tensor->numel());
    regions[i] = std::move(reg);
  }

  auto status = worker_ptr_->PullDense(regions.data(), regions.size(), tid);
  pull_dense_status->push_back(std::move(status));
}

void FleetWrapper::PullDenseVarsSync(
    const Scope& scope, const uint64_t tid,
    const std::vector<std::string>& var_names) {
  auto& regions = regions_[tid];
  regions.clear();
  regions.reserve(var_names.size());
  for (auto& t : var_names) {
    Variable* var = scope.FindVar(t);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    if (!platform::is_gpu_place(tensor->place())) {
      float* w = tensor->data<float>();
      paddle::distributed::Region reg(w, tensor->numel());
      regions.emplace_back(std::move(reg));
    }
  }
  auto status = worker_ptr_->PullDense(regions.data(), regions.size(), tid);
  status.wait();
}

void FleetWrapper::PushDenseParamSync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<std::string>& var_names) {
  auto place = platform::CPUPlace();
  std::vector<paddle::distributed::Region> regions;
  for (auto& t : var_names) {
    Variable* var = scope.FindVar(t);
    CHECK(var != nullptr) << "var[" << t << "] not found";
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    if (!platform::is_gpu_place(tensor->place())) {
      float* g = tensor->mutable_data<float>(place);
      paddle::distributed::Region reg(g, tensor->numel());
      regions.emplace_back(std::move(reg));
    }
  }
  auto push_status =
      worker_ptr_->PushDenseParam(regions.data(), regions.size(), table_id);
  push_status.wait();
  auto status = push_status.get();
  CHECK(status == 0) << "push dense param failed, status[" << status << "]";
}

void FleetWrapper::PushDenseVarsSync(
    Scope* scope, const uint64_t table_id,
    const std::vector<std::string>& var_names) {}

void FleetWrapper::PushDenseVarsAsync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<std::string>& var_names,
    std::vector<std::future<int32_t>>* push_sparse_status, float scale_datanorm,
    int batch_size) {
  auto place = platform::CPUPlace();
  std::vector<paddle::distributed::Region> regions;
  for (auto& t : var_names) {
    Variable* var = scope.FindVar(t);
    CHECK(var != nullptr) << "var[" << t << "] not found";
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    int count = tensor->numel();
    float* g = tensor->mutable_data<float>(place);
    // TODO(zhaocaibei123): how to get batch_size in op?
    if (scale_datanorm >= 0) {
      if (t.find(".batch_size@GRAD") != std::string::npos ||
          t.find(".batch_sum@GRAD") != std::string::npos) {
        Eigen::Map<Eigen::MatrixXf> mat(g, 1, count);
        float scale = 1.0 / batch_size;
        mat *= scale;
      } else if (t.find(".batch_square_sum@GRAD") != std::string::npos) {
        VLOG(3) << "epsilon: " << scale_datanorm;
        for (int i = 0; i < count; ++i) {
          g[i] = (g[i] - batch_size * scale_datanorm) / batch_size +
                 batch_size * scale_datanorm;
        }
      }
    }

    paddle::distributed::Region reg(g, tensor->numel());
    regions.emplace_back(std::move(reg));
    VLOG(3) << "FleetWrapper::PushDenseVarsAsync Var " << t << " talbe_id "
            << table_id << " Temp_data[0] " << g[0] << " Temp_data[-1] "
            << g[tensor->numel() - 1];
  }

  auto push_status =
      worker_ptr_->PushDense(regions.data(), regions.size(), table_id);
}

void FleetWrapper::PushSparseVarsAsync(
    const Scope& scope, const uint64_t table_id,
    const std::string& grad_varname,
    std::vector<std::future<int32_t>>* push_sparse_status) {
  std::vector<std::string> varnames;
  varnames.push_back(grad_varname);

  auto* communicator = Communicator::GetInstance();
  PADDLE_ENFORCE_EQ(
      communicator->Check(table_id), true,
      platform::errors::InvalidArgument(
          "can not find table: %s, please check your config", table_id));
  communicator->Send(varnames, scope);
}

void FleetWrapper::PushSparseVarsWithLabelAsync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<uint64_t>& fea_keys, const std::vector<float>& fea_labels,
    const std::vector<std::string>& sparse_key_names,
    const std::vector<std::string>& sparse_grad_names, const int emb_dim,
    std::vector<std::vector<float>>* push_values,
    std::vector<std::future<int32_t>>* push_sparse_status, const int batch_size,
    const bool use_cvm, const bool dump_slot,
    std::vector<uint64_t>* sparse_push_keys, const bool no_cvm) {
  // not support
  return;
}

void FleetWrapper::PushSparseFromTensorWithLabelAsync(
    const Scope& scope, const uint64_t table_id, int fea_dim,
    uint64_t padding_id, bool scale_sparse, const std::string& accesor,
    const std::string& click_name, platform::Place place,
    const std::vector<std::string>& input_names,
    std::vector<const LoDTensor*>* inputs,
    std::vector<const LoDTensor*>* outputs) {
  // not support
  return;
}

void FleetWrapper::PushSparseFromTensorAsync(
    const uint64_t table_id, int fea_dim, uint64_t padding_id,
    platform::Place place, std::vector<const LoDTensor*>* inputs,
    const LoDTensor* shows, const LoDTensor* clks,
    std::vector<LoDTensor*>* outputs, bool use_cvm_op) {
  int batch_size = -1;
  bool batch_size_consist = true;
  for (auto* input : *inputs) {
    int cur_batch_size =
        input->lod().size() ? input->lod()[0].size() - 1 : input->dims()[0];
    if (batch_size == -1) {
      batch_size = cur_batch_size;
    } else if (batch_size != cur_batch_size) {
      // CHECK(batch_size == cur_batch_size);  // NOLINT
      batch_size_consist = false;
      break;
    }
  }
  CHECK(batch_size > 0);  // NOLINT

  int show_size =
      shows->lod().size() ? shows->lod()[0].size() - 1 : shows->dims()[0];
  CHECK(show_size == batch_size || show_size == 1);
  int clk_size =
      clks->lod().size() ? clks->lod()[0].size() - 1 : clks->dims()[0];
  CHECK(clk_size == batch_size || clk_size == 1);

  CHECK(outputs->size() == inputs->size());
  std::vector<uint64_t> push_keys;
  push_keys.reserve(MAX_FEASIGN_NUM / 100);
  std::vector<std::vector<float>> push_values;
  push_values.reserve(MAX_FEASIGN_NUM / 100);
  size_t output_len = 0;
  size_t input_idx = 0;

  VLOG(2) << "fleet.cc::emb_dim: " << fea_dim;

  // TODO(zhaocaibei123): check type of show/clk is int? float? uint64?
  // const long int* show_tensor = shows->data<int64_t>();
  // const long int* clk_tensor = clks->data<int64_t>();
  const int64_t* show_tensor = shows->data<int64_t>();
  const int64_t* clk_tensor = clks->data<int64_t>();

  for (size_t index = 0; index < inputs->size(); ++index) {
    framework::LoDTensor* g_tensor = outputs->at(index);
    float* g = g_tensor->data<float>();
    // no cvm
    if (batch_size_consist) {  // TODO(zhaocaibei123): add config
                               // scale_sparse_gradient_with_batch_size_
      Eigen::Map<
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          g_mat(g, g_tensor->numel() / fea_dim, fea_dim);
      if (use_cvm_op) {
        g_mat.rightCols(fea_dim - 2) *= batch_size;
      } else {
        g_mat.rightCols(fea_dim) *= batch_size;
      }
    }

    const framework::LoDTensor* tensor = inputs->at(index);
    const int64_t* ids = tensor->data<int64_t>();
    size_t len = tensor->numel();
    output_len = 0;

    if (tensor->lod().size() > 0) {
      for (size_t i = 0; i < tensor->lod()[0].size() - 1; ++i) {
        for (int j = tensor->lod()[0][i]; j < tensor->lod()[0][i + 1];
             ++j, output_len += fea_dim) {
          uint64_t real_id = static_cast<uint64_t>(ids[j]);
          if (real_id == padding_id) {
            continue;
          }
          push_keys.emplace_back(real_id);
          if (use_cvm_op) {
            push_values.emplace_back(fea_dim + 1);
            push_values.back()[0] = 2;  // TODO(zhaocaibei123): slot
            float* data = push_values.back().data() + 1;
            memcpy(data, g + output_len, sizeof(float) * fea_dim);
          } else {
            push_values.emplace_back(fea_dim + 3);
            // slot show clk grad... consistent with CtrCommonPushValue defined
            // in
            // ctr_accessor.h
            push_values.back()[0] = 2;  // TODO(zhaocaibei123): slot
            push_values.back()[1] =
                (i >= show_size ? 1 : static_cast<float>(show_tensor[i]));
            push_values.back()[2] =
                (i >= clk_size ? 0 : static_cast<float>(clk_tensor[i]));
            float* data = push_values.back().data() + 3;
            memcpy(data, g + output_len, sizeof(float) * fea_dim);
          }
          ++input_idx;
        }
      }
    } else {
      for (size_t i = 0; i < len; ++i, output_len += fea_dim) {
        uint64_t real_id = static_cast<uint64_t>(ids[i]);
        if (real_id == padding_id) {
          continue;
        }
        push_keys.emplace_back(real_id);
        if (use_cvm_op) {
          push_values.emplace_back(fea_dim + 1);
          push_values.back()[0] = 2;  // TODO(zhaocaibei123): slot
          float* data = push_values.back().data() + 1;
          memcpy(data, g + output_len, sizeof(float) * fea_dim);
        } else {
          push_values.emplace_back(fea_dim + 3);
          // slot show clk grad... consistent with CtrCommonPushValue defined in
          // ctr_accessor.h
          push_values.back()[0] = 2;  // TODO(zhaocaibei123): slot
          push_values.back()[1] =
              (i >= show_size ? 1 : static_cast<float>(show_tensor[i]));
          push_values.back()[2] =
              (i >= clk_size ? 0 : static_cast<float>(clk_tensor[i]));
          float* data = push_values.back().data() + 3;
          memcpy(data, g + output_len, sizeof(float) * fea_dim);
        }
        ++input_idx;
      }
    }
    CHECK(output_len == g_tensor->numel());
  }

  std::vector<float*> push_g_vec(input_idx, nullptr);

  for (auto i = 0u; i < push_keys.size(); ++i) {
    push_g_vec[i] = push_values.at(i).data();
  }

  auto status = worker_ptr_->PushSparse(table_id, push_keys.data(),
                                        (const float**)push_g_vec.data(),
                                        push_keys.size());
}

void FleetWrapper::LoadModel(const std::string& path, const int mode) {
  auto ret = worker_ptr_->Load(path, std::to_string(mode));
  ret.wait();
  if (ret.get() != 0) {
    LOG(ERROR) << "load model from path:" << path << " failed";
  }
}

void FleetWrapper::LoadModelOneTable(const uint64_t table_id,
                                     const std::string& path, const int mode) {
  auto ret = worker_ptr_->Load(table_id, path, std::to_string(mode));
  ret.wait();
  if (ret.get() != 0) {
    LOG(ERROR) << "load model of table id: " << table_id
               << ", from path: " << path << " failed";
  }
}

void FleetWrapper::SaveModel(const std::string& path, const int mode) {
  auto ret = worker_ptr_->Save(path, std::to_string(mode));
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "save model failed";
  }
}

void FleetWrapper::SaveModelOneTable(const uint64_t table_id,
                                     const std::string& path, const int mode) {
  auto ret = worker_ptr_->Save(table_id, path, std::to_string(mode));
  ret.wait();
  if (ret.get() != 0) {
    LOG(ERROR) << "save model of table id: " << table_id
               << ", to path: " << path << " failed";
  }
}

void FleetWrapper::RecvAndSaveTable(const uint64_t table_id,
                                    const std::string& path) {
  auto ret = worker_ptr_->RecvAndSaveTable(table_id, path);
  if (ret != 0) {
    LOG(ERROR) << "save model of table id: " << table_id
               << ", to path: " << path << " failed";
  }
}

void FleetWrapper::PrintTableStat(const uint64_t table_id) {
  auto ret = worker_ptr_->PrintTableStat(table_id);
  ret.wait();
  int32_t err_code = ret.get();
  if (err_code == -1) {
    LOG(ERROR) << "print table stat failed";
  }
}

void FleetWrapper::ShrinkSparseTable(int table_id, int threshold) {
  auto ret = worker_ptr_->Shrink(table_id, std::to_string(threshold));
  ret.wait();
  int32_t err_code = ret.get();
  if (err_code == -1) {
    LOG(ERROR) << "shrink sparse table stat failed";
  }
}

void FleetWrapper::ClearModel() {
  auto ret = pserver_ptr_->_worker_ptr->Clear();
  ret.wait();
}

void FleetWrapper::ClearOneTable(const uint64_t table_id) {
  auto ret = pserver_ptr_->_worker_ptr->Clear(table_id);
  ret.wait();
}

void FleetWrapper::ShrinkDenseTable(int table_id, Scope* scope,
                                    std::vector<std::string> var_list,
                                    float decay, int emb_dim) {
  std::vector<paddle::distributed::Region> regions;
  for (std::string& name : var_list) {
    if (name.find("batch_sum") != std::string::npos) {
      Variable* var = scope->FindVar(name);
      CHECK(var != nullptr) << "var[" << name << "] not found";
      VLOG(3) << "prepare shrink dense batch_sum";
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      float* g = tensor->data<float>();

      // show_batch_sum += N * log(decay)
      std::string size_name = name;
      size_name.replace(size_name.find("batch_sum"), size_name.length(),
                        "batch_size");
      Variable* var_size = scope->FindVar(size_name);
      CHECK(var_size != nullptr) << "var[" << size_name << "] not found";
      VLOG(3) << "shrink dense batch_sum: " << name << ", " << size_name;
      float* g_size = var_size->GetMutable<LoDTensor>()->data<float>();

      for (int k = 0; k < tensor->numel(); k += emb_dim) {
        g[k] = g[k] + g_size[k] * log(decay);
      }
      paddle::distributed::Region reg(g, tensor->numel());
      regions.emplace_back(std::move(reg));
    } else {
      Variable* var = scope->FindVar(name);
      CHECK(var != nullptr) << "var[" << name << "] not found";
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      float* g = tensor->data<float>();
      paddle::distributed::Region reg(g, tensor->numel());
      regions.emplace_back(std::move(reg));
    }
  }
  auto push_status = pserver_ptr_->_worker_ptr->PushDenseParam(
      regions.data(), regions.size(), table_id);
  push_status.wait();
  auto status = push_status.get();
  if (status != 0) {
    // PADDLE_THORW(platform::errors::Fatal(
    //    "push shrink dense param failed, status is [%d].", status));
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
}

void FleetWrapper::ClientFlush() {
  if (worker_ptr_.get() == nullptr) {
    VLOG(0) << "worker_ptr null, do nothing";
    return;
  }
  auto ret = worker_ptr_->Flush();
  ret.wait();
  int32_t err_code = ret.get();
  if (err_code == -1) {
    LOG(ERROR) << "Client Flush failed";
  }
}

int FleetWrapper::RegisterClientToClientMsgHandler(int msg_type,
                                                   MsgHandlerFunc handler) {
  if (worker_ptr_.get() == nullptr) {
    VLOG(0) << "FleetWrapper::Client is null";
    return -1;
  } else {
    return worker_ptr_->RegisteClient2ClientMsgHandler(msg_type, handler);
  }
}

std::future<int32_t> FleetWrapper::SendClientToClientMsg(
    int msg_type, int to_client_id, const std::string& msg) {
  return worker_ptr_->SendClient2ClientMsg(msg_type, to_client_id, msg);
}

double FleetWrapper::GetCacheThreshold(int table_id) {
  double cache_threshold = 0.0;
  auto ret = worker_ptr_->Flush();
  ret.wait();
  ret = worker_ptr_->GetCacheThreshold(table_id, cache_threshold);
  ret.wait();
  if (cache_threshold < 0) {
    LOG(ERROR) << "get cache threshold failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
  return cache_threshold;
}

void FleetWrapper::CacheShuffle(int table_id, const std::string& path,
                                const int mode, const double cache_threshold) {
  auto ret = worker_ptr_->CacheShuffle(table_id, path, std::to_string(mode),
                                       std::to_string(cache_threshold));
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "cache shuffle failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
}

int32_t FleetWrapper::SaveCache(int table_id, const std::string& path,
                                const int mode) {
  auto ret = worker_ptr_->SaveCache(table_id, path, std::to_string(mode));
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "table save cache failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
  return feasign_cnt;
}

std::default_random_engine& FleetWrapper::LocalRandomEngine() {
  struct engine_wrapper_t {
    std::default_random_engine engine;

    engine_wrapper_t() {
      struct timespec tp;
      clock_gettime(CLOCK_REALTIME, &tp);
      double cur_time = tp.tv_sec + tp.tv_nsec * 1e-9;
      static std::atomic<uint64_t> x(0);
      std::seed_seq sseq = {x++, x++, x++, (uint64_t)(cur_time * 1000)};
      engine.seed(sseq);
    }
  };
  thread_local engine_wrapper_t r;
  return r.engine;
}

size_t FleetWrapper::GetAbsoluteSum(size_t start, size_t end, size_t level,
                                    const framework::LoD& lod) {
  if (level >= lod.size() - 1) {
    return end - start;
  }
  size_t ret = 0;
  for (size_t i = start; i < end - 1; ++i) {
    size_t pos1 = lod[level][i];
    size_t pos2 = lod[level][i + 1];
    ret += GetAbsoluteSum(pos1, pos2, level + 1, lod);
  }
  return ret;
}

}  // end namespace distributed
}  // end namespace paddle
