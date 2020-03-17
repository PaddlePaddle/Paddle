// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include <algorithm>
#include <utility>
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {

const uint32_t MAX_FEASIGN_NUM = 1024 * 100 * 100;
std::shared_ptr<FleetWrapper> FleetWrapper::s_instance_ = NULL;
bool FleetWrapper::is_initialized_ = false;

#ifdef PADDLE_WITH_PSLIB
std::shared_ptr<paddle::distributed::PSlib> FleetWrapper::pslib_ptr_ = NULL;
#endif

void FleetWrapper::SetClient2ClientConfig(int request_timeout_ms,
                                          int connect_timeout_ms,
                                          int max_retry) {
  client2client_request_timeout_ms_ = request_timeout_ms;
  client2client_connect_timeout_ms_ = connect_timeout_ms;
  client2client_max_retry_ = max_retry;
}

void FleetWrapper::InitServer(const std::string& dist_desc, int index) {
#ifdef PADDLE_WITH_PSLIB
  if (!is_initialized_) {
    VLOG(3) << "Going to init server";
    pslib_ptr_ = std::shared_ptr<paddle::distributed::PSlib>(
        new paddle::distributed::PSlib());
    pslib_ptr_->init_server(dist_desc, index);
    is_initialized_ = true;
  } else {
    VLOG(3) << "Server can be initialized only once";
  }
#endif
}

void FleetWrapper::InitWorker(const std::string& dist_desc,
                              const std::vector<uint64_t>& host_sign_list,
                              int node_num, int index) {
#ifdef PADDLE_WITH_PSLIB
  if (!is_initialized_) {
    VLOG(3) << "Going to init worker";
    pslib_ptr_ = std::shared_ptr<paddle::distributed::PSlib>(
        new paddle::distributed::PSlib());
    pslib_ptr_->init_worker(dist_desc,
                            const_cast<uint64_t*>(host_sign_list.data()),
                            node_num, index);
    is_initialized_ = true;
  } else {
    VLOG(3) << "Worker can be initialized only once";
  }
#endif
}

void FleetWrapper::StopServer() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to stop server";
  pslib_ptr_->stop_server();
#endif
}

void FleetWrapper::FinalizeWorker() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to finalize worker";
  pslib_ptr_->finalize_worker();
#endif
}

uint64_t FleetWrapper::RunServer() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to run server";
  return pslib_ptr_->run_server();
#else
  return 0;
#endif
}

uint64_t FleetWrapper::RunServer(const std::string& ip, uint32_t port) {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to run server with ip " << ip << " port " << port;
  auto ret = pslib_ptr_->run_server(ip, port);
  return ret;
#else
  return 0;
#endif
}

void FleetWrapper::GatherServers(const std::vector<uint64_t>& host_sign_list,
                                 int node_num) {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to gather server ips";
  pslib_ptr_->gather_servers(const_cast<uint64_t*>(host_sign_list.data()),
                             node_num);
#endif
}

void FleetWrapper::GatherClients(const std::vector<uint64_t>& host_sign_list) {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to gather client ips";
  size_t len = host_sign_list.size();
  pslib_ptr_->gather_clients(const_cast<uint64_t*>(host_sign_list.data()), len);
#endif
}

std::vector<uint64_t> FleetWrapper::GetClientsInfo() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to get client info";
  return pslib_ptr_->get_client_info();
#endif
  return std::vector<uint64_t>();
}

void FleetWrapper::CreateClient2ClientConnection() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to create client2client connection";
  pslib_ptr_->create_client2client_connection(client2client_request_timeout_ms_,
                                              client2client_connect_timeout_ms_,
                                              client2client_max_retry_);
#endif
}

void FleetWrapper::PullSparseVarsSync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<std::string>& var_names, std::vector<uint64_t>* fea_keys,
    std::vector<std::vector<float>>* fea_values, int fea_value_dim,
    const std::vector<std::string>& var_emb_names) {
#ifdef PADDLE_WITH_PSLIB
  std::vector<::std::future<int32_t>> pull_sparse_status;
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
  auto status = pslib_ptr_->_worker_ptr->pull_sparse(
      pull_result_ptr.data(), table_id, fea_keys->data(), fea_keys->size());
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
#endif
}

void FleetWrapper::PullDenseVarsAsync(
    const Scope& scope, const uint64_t tid,
    const std::vector<std::string>& var_names,
    std::vector<::std::future<int32_t>>* pull_dense_status,
    std::vector<std::vector<float>>& dense_region,
    const paddle::platform::Place& place) {
#ifdef PADDLE_WITH_PSLIB
  auto& regions = _regions[tid];
  regions.clear();
  regions.resize(var_names.size());
  for (auto i = 0u; i < var_names.size(); ++i) {
    Variable* var = scope.FindVar(var_names[i]);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* w = tensor->data<float>();
    paddle::ps::Region reg;
    if (platform::is_cpu_place(place)) {
      reg = paddle::ps::Region(w, tensor->numel());
    }
    else {
      if (int(dense_region[i].size()) < tensor->numel()) {
        dense_region[i].resize(tensor->numel());
      }
      reg = paddle::ps::Region(dense_region[i].data(), tensor->numel());
    }
    regions[i] = std::move(reg);
  }
  auto status =
      pslib_ptr_->_worker_ptr->pull_dense(regions.data(), regions.size(), tid);
  pull_dense_status->push_back(std::move(status));
#endif
}

void FleetWrapper::PullDenseVarsSync(
    const Scope& scope, const uint64_t tid,
    const std::vector<std::string>& var_names) {
#ifdef PADDLE_WITH_PSLIB
  auto& regions = _regions[tid];
  regions.clear();
  regions.reserve(var_names.size());
  for (auto& t : var_names) {
    Variable* var = scope.FindVar(t);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* w = tensor->data<float>();
    paddle::ps::Region reg(w, tensor->numel());
    regions.emplace_back(std::move(reg));
  }
  auto status =
      pslib_ptr_->_worker_ptr->pull_dense(regions.data(), regions.size(), tid);
  status.wait();
#endif
}

void FleetWrapper::PushDenseParamSync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<std::string>& var_names) {
#ifdef PADDLE_WITH_PSLIB
  std::vector<std::vector<float>> bufs;
  bufs.resize(var_names.size());
  std::vector<paddle::ps::Region> regions;
  for (size_t i = 0; i < var_names.size(); ++i) {
    auto& t = var_names[i];
    Variable* var = scope.FindVar(t);
    CHECK(var != nullptr) << "var[" << t << "] not found";
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* t_data = tensor->data<float>();
    float* g = t_data;
    int count = tensor->numel();
    #ifdef PADDLE_WITH_CUDA
    if (!platform::is_cpu_place(tensor->place())) {
      auto& buf = bufs[i];
      if (int(buf.size()) < count) {
        buf.resize(count);
      }
      memory::Copy(
          platform::CPUPlace(),
          buf.data(),
          boost::get<platform::CUDAPlace>(tensor->place()),
          t_data, sizeof(float) * count, nullptr);
      g = buf.data();
    }
    #endif
    paddle::ps::Region reg(g, count);
    regions.emplace_back(std::move(reg));
    
  }
  auto push_status = pslib_ptr_->_worker_ptr->push_dense_param(
      regions.data(), regions.size(), table_id);
  push_status.wait();
  auto status = push_status.get();
  CHECK(status == 0) << "push dense param failed, status[" << status << "]";
#endif
}

void FleetWrapper::PushDenseVarsSync(
    Scope* scope, const uint64_t table_id,
    const std::vector<std::string>& var_names) {}

void FleetWrapper::PushDenseVarsAsync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<std::string>& var_names,
    std::vector<::std::future<int32_t>>* push_sparse_status,
    float scale_datanorm, int batch_size,
    std::vector<std::vector<float>>& dense_grad_regions_,
    const paddle::platform::Place& place,
    cudaStream_t stream,
    cudaEvent_t event) {
#ifdef PADDLE_WITH_PSLIB
  if (!platform::is_cpu_place(place)) {
    //platform::DeviceContextPool::Instance().Get(place)->Wait();
  }
  std::vector<paddle::ps::Region> regions;
  for (size_t i = 0; i < var_names.size(); ++i) {
    auto& t = var_names[i];
    Variable* var = scope.FindVar(t);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    int count = tensor->numel();

    float* g_data = tensor->data<float>();
    float* g = g_data;
    #ifdef PADDLE_WITH_CUDA
    if (!platform::is_cpu_place(place)) {
      Variable *pin_var = scope.FindVar(t + "pin");
      LoDTensor* pin_tensor = pin_var->GetMutable<LoDTensor>();
      float *pin_g = pin_tensor->mutable_data<float>(tensor->dims(), platform::CUDAPinnedPlace());
      //if (int(dense_grad_regions_[i].size()) < count) {
      //  dense_grad_regions_[i].resize(count);
      //}
      memory::Copy(
          platform::CUDAPinnedPlace(),
          pin_g,
          boost::get<platform::CUDAPlace>(place),
          g_data, sizeof(float) * count, stream);
      g = pin_g;
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(event, stream));
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamWaitEvent(stream, event, 0));
    }
    #endif
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
    paddle::ps::Region reg(g, count);
    regions.emplace_back(std::move(reg));
  }
  auto status = pslib_ptr_->_worker_ptr->push_dense(regions.data(),
                                                    regions.size(), table_id);
  push_sparse_status->push_back(std::move(status));
#endif
}

void FleetWrapper::PushSparseVarsWithLabelAsync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<uint64_t>& fea_keys, const std::vector<float>& fea_labels,
    const std::vector<std::string>& sparse_key_names,
    const std::vector<std::string>& sparse_grad_names, const int emb_dim,
    std::vector<std::vector<float>>* push_values,
    std::vector<::std::future<int32_t>>* push_sparse_status,
    const int batch_size, const bool use_cvm, const bool dump_slot,
    std::vector<uint64_t>* sparse_push_keys, const bool no_cvm,
    const paddle::platform::Place& place,
    std::vector<float>& sparse_grad_region,
    cudaStream_t stream,
    cudaEvent_t event) {
#ifdef PADDLE_WITH_PSLIB
  int offset = 2;
  int slot_offset = 0;
  int grad_dim = emb_dim;
  int show_index = 0;
  int click_index = 1;
  bool in_cpu = true;
  if (use_cvm) {
    offset = 0;
    grad_dim = emb_dim - 2;
  }
  if (no_cvm) {
    offset = 0;
    grad_dim = emb_dim;
  }
  if (dump_slot) {
    slot_offset = 1;
    show_index = 1;
    click_index = 2;
  }
  CHECK_GE(grad_dim, 0);

  sparse_push_keys->clear();
  sparse_push_keys->reserve(fea_keys.size() + 1);
  push_values->resize(fea_keys.size() + 1);
  for (auto& t : *push_values) {
    t.resize(emb_dim + offset + slot_offset);
  }
  uint64_t fea_idx = 0u;
  if (!platform::is_cpu_place(place)) {
    
    in_cpu = false;
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamWaitEvent(stream, event, 0));
    
    Variable* g_var = scope.FindVar(sparse_grad_names[0]);
    LoDTensor* g_tensor = g_var->GetMutable<LoDTensor>();
    float* g = g_tensor->data<float>();
    
    Variable *pin_var = scope.FindVar(sparse_grad_names[0] + "pin");
    LoDTensor* pin_tensor = pin_var->GetMutable<LoDTensor>();
    float *pin_g = pin_tensor->mutable_data<float>(g_tensor->dims(), platform::CUDAPinnedPlace());
    #ifdef PADDLE_WITH_CUDA
    memory::Copy(
        platform::CUDAPinnedPlace(),
        pin_g,
        boost::get<platform::CUDAPlace>(place),
        g, sizeof(float) * g_tensor->numel(), stream);
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(event, stream));
    #endif
  }
  for (size_t i = 0;
       i < sparse_key_names.size() && i < sparse_grad_names.size(); ++i) {
    Variable* var = scope.FindVar(sparse_key_names[i]);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    if (tensor == nullptr) {
      LOG(ERROR) << "tensor of var[" << sparse_key_names[i] << "] is null";
      exit(-1);
    }
    size_t len = tensor->numel();
    int64_t* ids = tensor->data<int64_t>();
    int slot = 0;
    if (dump_slot) {
      slot = boost::lexical_cast<int>(sparse_key_names[i]);
    }
    Variable* g_var = scope.FindVar(sparse_grad_names[i]);
    if (g_var == nullptr) {
      continue;
    }
    LoDTensor* g_tensor = g_var->GetMutable<LoDTensor>();
    if (g_tensor == nullptr) {
      LOG(ERROR) << "tensor of var[" << sparse_key_names[i] << "] is null";
      exit(-1);
    }
    float* g = g_tensor->data<float>();
    if (!in_cpu) {
      //if (sparse_grad_region.size() < (len * emb_dim)) {
      //  sparse_grad_region.resize(len * emb_dim);
      //}
      
      Variable *pin_var = scope.FindVar(sparse_grad_names[i] + "pin");
      LoDTensor* pin_tensor = pin_var->GetMutable<LoDTensor>();
      g = pin_tensor->data<float>();
      
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamWaitEvent(stream, event, 0));
      if (i + 1 < sparse_grad_names.size()) {
    
        Variable* next_g_var = scope.FindVar(sparse_grad_names[i + 1]);
        LoDTensor* next_g_tensor = next_g_var->GetMutable<LoDTensor>();
        float* next_g = next_g_tensor->data<float>();
        
        Variable *next_pin_var = scope.FindVar(sparse_grad_names[i + 1] + "pin");
        LoDTensor* next_pin_tensor = next_pin_var->GetMutable<LoDTensor>();
        float * next_pin_g = next_pin_tensor->mutable_data<float>(next_g_tensor->dims(), platform::CUDAPinnedPlace());
        #ifdef PADDLE_WITH_CUDA
        memory::Copy(
            platform::CUDAPinnedPlace(),
            next_pin_g,
            boost::get<platform::CUDAPlace>(place),
            next_g, sizeof(float) * next_g_tensor->numel(), stream);
        PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventRecord(event, stream));
        #endif
      }
    }

    if (scale_sparse_gradient_with_batch_size_ && grad_dim > 0) {
      int dim = emb_dim + offset;
      Eigen::Map<
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          g_mat(g, g_tensor->numel() / dim, dim);
      g_mat.rightCols(grad_dim) *= batch_size;
    }
    for (auto id_idx = 0u; id_idx < len; ++id_idx) {
      if (ids[id_idx] == 0) {
        g += emb_dim;
        continue;
      }
      sparse_push_keys->push_back(ids[id_idx]);
      CHECK(fea_idx < (*push_values).size());

      if (use_cvm || no_cvm) {
        memcpy((*push_values)[fea_idx].data() + offset + slot_offset, g,
               sizeof(float) * emb_dim);
      } else {
        CHECK(fea_idx < fea_labels.size());
        memcpy((*push_values)[fea_idx].data() + offset + slot_offset, g,
               sizeof(float) * emb_dim);
        (*push_values)[fea_idx][show_index] = 1.0f;
        (*push_values)[fea_idx][click_index] =
            static_cast<float>(fea_labels[fea_idx]);
      }
      if (dump_slot) {
        (*push_values)[fea_idx][0] = static_cast<float>(slot);
      }
      g += emb_dim;
      fea_idx++;
    }
  }
  // slots whose embedding has been stop gradient or
  // not involved in forward-backward
  uint64_t no_grad_fea_num = 0u;
  for (size_t i = sparse_grad_names.size(); i < sparse_key_names.size(); ++i) {
    Variable* var = scope.FindVar(sparse_key_names[i]);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    if (tensor == nullptr) {
      LOG(ERROR) << "tensor of var[" << sparse_key_names[i] << "] is null";
      exit(-1);
    }
    size_t len = tensor->numel();
    int64_t* ids = tensor->data<int64_t>();
    for (auto id_idx = 0u; id_idx < len; ++id_idx) {
      if (ids[id_idx] == 0) {
        continue;
      }
      ++no_grad_fea_num;
    }
  }
  CHECK(fea_idx + no_grad_fea_num == fea_keys.size())
      << "fea_idx: " << fea_idx << " no_grad_fea_num: " << no_grad_fea_num
      << " features size: " << fea_keys.size();
  CHECK(fea_idx == sparse_push_keys->size());
  if (fea_idx == 0) {
    return;
  }
  std::vector<float*> push_g_vec;
  for (auto i = 0u; i < sparse_push_keys->size(); ++i) {
    push_g_vec.push_back((*push_values)[i].data());
  }
  auto status = pslib_ptr_->_worker_ptr->push_sparse(
      table_id, sparse_push_keys->data(), (const float**)push_g_vec.data(),
      sparse_push_keys->size());
  push_sparse_status->push_back(std::move(status));
#endif
}

void FleetWrapper::LoadFromPaddleModel(Scope& scope, const uint64_t table_id,
                                       std::vector<std::string> var_list,
                                       std::string model_path,
                                       std::string model_proto_file,
                                       std::vector<std::string> table_var_list,
                                       bool load_combine) {
#ifdef PADDLE_WITH_PSLIB
  // load ProgramDesc from model file
  auto read_proto_func = [](const std::string& filename) -> ProgramDesc {
    std::string contents;
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    fin.seekg(0, std::ios::end);
    contents.resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&contents[0], contents.size());
    fin.close();
    ProgramDesc program_desc(contents);
    return program_desc;
  };
  const ProgramDesc old_program = read_proto_func(model_proto_file);
  Scope* old_scope = new Scope();
  auto& old_block = old_program.Block(0);
  auto place = platform::CPUPlace();
  std::vector<std::string> old_param_list;

  for (auto& t : var_list) {
    VarDesc* old_var_desc = old_block.FindVar(t);
    if (old_var_desc == nullptr) {
      continue;
    }
    // init variable in scope
    Variable* old_var = old_scope->Var(old_var_desc->Name());
    InitializeVariable(old_var, old_var_desc->GetType());
    old_param_list.push_back(t);
    if (load_combine) {
      continue;
    }
    // load variable from model
    paddle::framework::AttributeMap attrs;
    attrs.insert({"file_path", model_path + "/" + old_var_desc->Name()});
    auto load_op = paddle::framework::OpRegistry::CreateOp(
        "load", {}, {{"Out", {old_var_desc->Name()}}}, attrs);
    load_op->Run(*old_scope, place);
  }

  if (load_combine) {
    std::sort(old_param_list.begin(), old_param_list.end());
    paddle::framework::AttributeMap attrs;
    attrs.insert({"file_path", model_path});
    auto load_op = paddle::framework::OpRegistry::CreateOp(
        "load_combine", {}, {{"Out", old_param_list}}, attrs);
    load_op->Run(*old_scope, place);
  }

  for (auto& t : old_param_list) {
    Variable* old_var = old_scope->Var(t);
    // old model data, here we assume data type is float
    LoDTensor* old_tensor = old_var->GetMutable<LoDTensor>();
    float* old_data = old_tensor->data<float>();
    // new model data, here we assume data type is float
    Variable* var = scope.FindVar(t);
    CHECK(var != nullptr) << "var[" << t << "] not found";
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* data = tensor->data<float>();
    // copy from old data to new data
    if (old_tensor->numel() > tensor->numel()) {
      memcpy(data, old_data, tensor->numel() * sizeof(float));
    } else {
      memcpy(data, old_data, old_tensor->numel() * sizeof(float));
    }
  }
  delete old_scope;
  PushDenseParamSync(scope, table_id, table_var_list);
#endif
}

void FleetWrapper::LoadModel(const std::string& path, const int mode) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->load(path, std::to_string(mode));
  ret.wait();
  if (ret.get() != 0) {
    LOG(ERROR) << "load model from path:" << path << " failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::LoadModel does nothing when no pslib";
#endif
}

void FleetWrapper::LoadModelOneTable(const uint64_t table_id,
                                     const std::string& path, const int mode) {
#ifdef PADDLE_WITH_PSLIB
  auto ret =
      pslib_ptr_->_worker_ptr->load(table_id, path, std::to_string(mode));
  ret.wait();
  if (ret.get() != 0) {
    LOG(ERROR) << "load model of table id: " << table_id
               << ", from path: " << path << " failed";
  }
#else
  VLOG(0) << "FleetWrapper::LoadModel does nothing when no pslib";
#endif
}

void FleetWrapper::SaveModel(const std::string& path, const int mode) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->save(path, std::to_string(mode));
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "save model failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::SaveModel does nothing when no pslib";
#endif
}

void FleetWrapper::PrintTableStat(const uint64_t table_id) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->print_table_stat(table_id);
  ret.wait();
  int32_t err_code = ret.get();
  if (err_code == -1) {
    LOG(ERROR) << "print table stat failed";
  }
#else
  VLOG(0) << "FleetWrapper::PrintTableStat does nothing when no pslib";
#endif
}

double FleetWrapper::GetCacheThreshold(int table_id) {
#ifdef PADDLE_WITH_PSLIB
  double cache_threshold = 0.0;
  auto ret = pslib_ptr_->_worker_ptr->flush();
  ret.wait();
  ret = pslib_ptr_->_worker_ptr->get_cache_threshold(table_id, cache_threshold);
  ret.wait();
  if (cache_threshold < 0) {
    LOG(ERROR) << "get cache threshold failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
  return cache_threshold;
#else
  VLOG(0) << "FleetWrapper::GetCacheThreshold does nothing when no pslib";
  return 0.0;
#endif
}

void FleetWrapper::CacheShuffle(int table_id, const std::string& path,
                                const int mode, const double cache_threshold) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->cache_shuffle(
      table_id, path, std::to_string(mode), std::to_string(cache_threshold));
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "cache shuffle failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::CacheShuffle does nothing when no pslib";
#endif
}

int32_t FleetWrapper::SaveCache(int table_id, const std::string& path,
                                const int mode) {
#ifdef PADDLE_WITH_PSLIB
  auto ret =
      pslib_ptr_->_worker_ptr->save_cache(table_id, path, std::to_string(mode));
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "table save cache failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
  return feasign_cnt;
#else
  VLOG(0) << "FleetWrapper::SaveCache does nothing when no pslib";
  return -1;
#endif
}

void FleetWrapper::ShrinkSparseTable(int table_id) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->shrink(table_id);
  ret.wait();
#else
  VLOG(0) << "FleetWrapper::ShrinkSparseTable does nothing when no pslib";
#endif
}

void FleetWrapper::ClearModel() {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->clear();
  ret.wait();
#else
  VLOG(0) << "FleetWrapper::ClearModel does nothing when no pslib";
#endif
}

void FleetWrapper::ShrinkDenseTable(int table_id, Scope* scope,
                                    std::vector<std::string> var_list,
                                    float decay, int emb_dim) {
#ifdef PADDLE_WITH_PSLIB
  std::vector<paddle::ps::Region> regions;
  for (std::string& name : var_list) {
    if (name.find("batch_sum") != std::string::npos) {
      Variable* var = scope->FindVar(name);
      CHECK(var != nullptr) << "var[" << name << "] not found";
      VLOG(0) << "prepare shrink dense batch_sum";
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
      paddle::ps::Region reg(g, tensor->numel());
      regions.emplace_back(std::move(reg));
    } else {
      Variable* var = scope->FindVar(name);
      CHECK(var != nullptr) << "var[" << name << "] not found";
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      float* g = tensor->data<float>();
      paddle::ps::Region reg(g, tensor->numel());
      regions.emplace_back(std::move(reg));
    }
  }
  auto push_status = pslib_ptr_->_worker_ptr->push_dense_param(
      regions.data(), regions.size(), table_id);
  push_status.wait();
  auto status = push_status.get();
  if (status != 0) {
    LOG(FATAL) << "push shrink dense param failed, status[" << status << "]";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::ShrinkSparseTable does nothing when no pslib";
#endif
}

void FleetWrapper::ClientFlush() {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->flush();
  ret.wait();
#else
  VLOG(0) << "FleetWrapper::ServerFlush does nothing when no pslib";
#endif
}

int FleetWrapper::RegisterClientToClientMsgHandler(int msg_type,
                                                   MsgHandlerFunc handler) {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "calling FleetWrapper::RegisterClientToClientMsgHandler";
  VLOG(3) << "pslib_ptr_=" << pslib_ptr_;
  VLOG(3) << "_worker_ptr=" << pslib_ptr_->_worker_ptr;
  return pslib_ptr_->_worker_ptr->registe_client2client_msg_handler(msg_type,
                                                                    handler);
#else
  VLOG(0) << "FleetWrapper::RegisterClientToClientMsgHandler"
          << " does nothing when no pslib";
#endif
  return 0;
}

std::future<int32_t> FleetWrapper::SendClientToClientMsg(
    int msg_type, int to_client_id, const std::string& msg) {
#ifdef PADDLE_WITH_PSLIB
  return pslib_ptr_->_worker_ptr->send_client2client_msg(msg_type, to_client_id,
                                                         msg);
#else
  VLOG(0) << "FleetWrapper::SendClientToClientMsg"
          << " does nothing when no pslib";
#endif
  return std::future<int32_t>();
}

std::default_random_engine& FleetWrapper::LocalRandomEngine() {
  struct engine_wrapper_t {
    std::default_random_engine engine;
#ifdef PADDLE_WITH_PSLIB
    engine_wrapper_t() {
      struct timespec tp;
      clock_gettime(CLOCK_REALTIME, &tp);
      double cur_time = tp.tv_sec + tp.tv_nsec * 1e-9;
      static std::atomic<uint64_t> x(0);
      std::seed_seq sseq = {x++, x++, x++, (uint64_t)(cur_time * 1000)};
      engine.seed(sseq);
    }
#endif
  };
  thread_local engine_wrapper_t r;
  return r.engine;
}

int32_t FleetWrapper::CopyTable(const uint64_t src_table_id,
                                const uint64_t dest_table_id) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->copy_table(src_table_id, dest_table_id);
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "copy table failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
  return feasign_cnt;
#else
  VLOG(0) << "FleetWrapper::CopyTable does nothing when no pslib";
  return 0;
#endif
}

int32_t FleetWrapper::CopyTableByFeasign(
    const uint64_t src_table_id, const uint64_t dest_table_id,
    const std::vector<uint64_t>& feasign_list) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->copy_table_by_feasign(
      src_table_id, dest_table_id, feasign_list.data(), feasign_list.size());
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "copy table by feasign failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
  return feasign_cnt;
#else
  VLOG(0) << "FleetWrapper::CopyTableByFeasign does nothing when no pslib";
  return 0;
#endif
}

}  // end namespace framework
}  // end namespace paddle
