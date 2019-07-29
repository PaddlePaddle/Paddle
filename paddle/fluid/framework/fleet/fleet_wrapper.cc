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
template <class AR>
paddle::ps::Archive<AR>& operator<<(paddle::ps::Archive<AR>& ar,
                                    const MultiSlotType& ins) {
  ar << ins.GetType();
  ar << ins.GetOffset();
  ar << ins.GetFloatData();
  ar << ins.GetUint64Data();
  return ar;
}

template <class AR>
paddle::ps::Archive<AR>& operator>>(paddle::ps::Archive<AR>& ar,
                                    MultiSlotType& ins) {
  ar >> ins.MutableType();
  ar >> ins.MutableOffset();
  ar >> ins.MutableFloatData();
  ar >> ins.MutableUint64Data();
  return ar;
}
#endif

#ifdef PADDLE_WITH_PSLIB
std::shared_ptr<paddle::distributed::PSlib> FleetWrapper::pslib_ptr_ = NULL;
#endif

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

uint64_t FleetWrapper::RunServer() {
#ifdef PADDLE_WITH_PSLIB
  VLOG(3) << "Going to run server";
  return pslib_ptr_->run_server();
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
  pslib_ptr_->create_client2client_connection();
#endif
}

void FleetWrapper::PullSparseVarsSync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<std::string>& var_names, std::vector<uint64_t>* fea_keys,
    std::vector<std::vector<float>>* fea_values, int fea_value_dim) {
#ifdef PADDLE_WITH_PSLIB
  std::vector<::std::future<int32_t>> pull_sparse_status;
  pull_sparse_status.resize(0);
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
    int len = tensor->numel();
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
      exit(-1);
    }
  }
#endif
}

void FleetWrapper::PullDenseVarsAsync(
    const Scope& scope, const uint64_t tid,
    const std::vector<std::string>& var_names,
    std::vector<::std::future<int32_t>>* pull_dense_status) {
#ifdef PADDLE_WITH_PSLIB
  auto& regions = _regions[tid];
  regions.clear();
  regions.resize(var_names.size());
  for (auto i = 0u; i < var_names.size(); ++i) {
    Variable* var = scope.FindVar(var_names[i]);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* w = tensor->data<float>();
    paddle::ps::Region reg(w, tensor->numel());
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
  auto place = platform::CPUPlace();
  std::vector<paddle::ps::Region> regions;
  for (auto& t : var_names) {
    Variable* var = scope.FindVar(t);
    CHECK(var != nullptr) << "var[" << t << "] not found";
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* g = tensor->mutable_data<float>(place);
    paddle::ps::Region reg(g, tensor->numel());
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
    float scale_datanorm, int batch_size) {
#ifdef PADDLE_WITH_PSLIB
  std::vector<paddle::ps::Region> regions;
  for (auto& t : var_names) {
    Variable* var = scope.FindVar(t);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    int count = tensor->numel();
    float* g = tensor->data<float>();
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
    const int batch_size, const bool use_cvm, const bool dump_slot) {
#ifdef PADDLE_WITH_PSLIB
  int offset = 2;
  int slot_offset = 0;
  int grad_dim = emb_dim;
  int show_index = 0;
  int click_index = 1;
  if (use_cvm) {
    offset = 0;
    grad_dim = emb_dim - 2;
  }
  if (dump_slot) {
    slot_offset = 1;
    show_index = 1;
    click_index = 2;
  }
  CHECK_GE(grad_dim, 0);

  push_values->resize(fea_keys.size() + 1);
  for (auto& t : *push_values) {
    t.resize(emb_dim + offset + slot_offset);
  }
  uint64_t fea_idx = 0u;
  for (size_t i = 0; i < sparse_key_names.size(); ++i) {
    Variable* var = scope.FindVar(sparse_key_names[i]);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    if (tensor == nullptr) {
      LOG(ERROR) << "tensor of var[" << sparse_key_names[i] << "] is null";
      exit(-1);
    }
    int len = tensor->numel();
    int64_t* ids = tensor->data<int64_t>();
    int slot = 0;
    if (dump_slot) {
      slot = boost::lexical_cast<int>(sparse_key_names[i]);
    }
    Variable* g_var = scope.FindVar(sparse_grad_names[i]);
    CHECK(g_var != nullptr) << "var[" << sparse_grad_names[i] << "] not found";
    LoDTensor* g_tensor = g_var->GetMutable<LoDTensor>();
    if (g_tensor == nullptr) {
      LOG(ERROR) << "tensor of var[" << sparse_key_names[i] << "] is null";
      exit(-1);
    }
    float* g = g_tensor->data<float>();

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
      CHECK(fea_idx < (*push_values).size());
      CHECK(fea_idx < fea_labels.size());

      if (use_cvm) {
        memcpy((*push_values)[fea_idx].data() + offset + slot_offset, g,
               sizeof(float) * emb_dim);
      } else {
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
  CHECK(fea_idx == fea_keys.size()) << "fea_idx: " << fea_idx
                                    << "features size: " << fea_keys.size();
  std::vector<float*> push_g_vec;
  for (auto i = 0u; i < fea_keys.size(); ++i) {
    push_g_vec.push_back((*push_values)[i].data());
  }
  auto status = pslib_ptr_->_worker_ptr->push_sparse(
      table_id, fea_keys.data(), (const float**)push_g_vec.data(),
      fea_keys.size());
  push_sparse_status->push_back(std::move(status));

#endif
}

void FleetWrapper::LoadFromPaddleModel(Scope& scope, const uint64_t table_id,
                                       std::vector<std::string> var_list,
                                       std::string model_path,
                                       std::string model_proto_file,
                                       bool load_combine) {
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
  PushDenseParamSync(scope, table_id, old_param_list);
}

void FleetWrapper::LoadModel(const std::string& path, const int mode) {
#ifdef PADDLE_WITH_PSLIB
  auto ret = pslib_ptr_->_worker_ptr->load(path, std::to_string(mode));
  ret.wait();
  if (ret.get() != 0) {
    LOG(ERROR) << "load model from path:" << path << " failed";
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
    exit(-1);
  }
#else
  VLOG(0) << "FleetWrapper::SaveModel does nothing when no pslib";
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

template <typename T>
void FleetWrapper::Serialize(const std::vector<T*>& t, std::string* str) {
#ifdef PADDLE_WITH_PSLIB
  paddle::ps::BinaryArchive ar;
  for (size_t i = 0; i < t.size(); ++i) {
    ar << *(t[i]);
  }
  *str = std::string(ar.buffer(), ar.length());
#else
  VLOG(0) << "FleetWrapper::Serialize does nothing when no pslib";
#endif
}

template <typename T>
void FleetWrapper::Deserialize(std::vector<T>* t, const std::string& str) {
#ifdef PADDLE_WITH_PSLIB
  if (str.length() == 0) {
    return;
  }
  paddle::ps::BinaryArchive ar;
  ar.set_read_buffer(const_cast<char*>(str.c_str()), str.length(), nullptr);
  if (ar.cursor() == ar.finish()) {
    return;
  }
  while (ar.cursor() < ar.finish()) {
    t->push_back(ar.get<T>());
  }
  CHECK(ar.cursor() == ar.finish());
  VLOG(3) << "Deserialize size " << t->size();
#else
  VLOG(0) << "FleetWrapper::Deserialize does nothing when no pslib";
#endif
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

template void FleetWrapper::Serialize<std::vector<MultiSlotType>>(
    const std::vector<std::vector<MultiSlotType>*>&, std::string*);
template void FleetWrapper::Deserialize<std::vector<MultiSlotType>>(
    std::vector<std::vector<MultiSlotType>>*, const std::string&);

}  // end namespace framework
}  // end namespace paddle
