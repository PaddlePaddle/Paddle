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

#pragma once

#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/communicator_common.h"
#include "paddle/fluid/distributed/service/service.h"
#include "paddle/fluid/framework/archive.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/framework/io/shell.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace framework {
class Scope;
class SelectedRows;
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace distributed {

class PSCore;

using framework::LoDTensor;
using framework::Scope;
using framework::SelectedRows;
using framework::Variable;

using RpcCtxMap = std::unordered_map<std::string, CommContext>;

class FleetWrapper {
 public:
  virtual ~FleetWrapper() {}
  FleetWrapper() {
    scale_sparse_gradient_with_batch_size_ = true;
    // trainer sleep some time for pserver core dump
    sleep_seconds_before_fail_exit_ = 300;
    // pserver request server timeout ms
    client2client_request_timeout_ms_ = 500000;
    // pserver connect server timeout_ms
    client2client_connect_timeout_ms_ = 10000;
    // pserver request max retry
    client2client_max_retry_ = 3;
  }

  // set client to client communication config
  void SetClient2ClientConfig(int request_timeout_ms, int connect_timeout_ms,
                              int max_retry);

  // Pull sparse variables from server in sync mode
  // Param<in>: scope, table_id, var_names, fea_keys, fea_dim, var_emb_names
  // Param<out>: fea_values
  void PullSparseVarsSync(const Scope& scope, const uint64_t table_id,
                          const std::vector<std::string>& var_names,
                          std::vector<uint64_t>* fea_keys,
                          std::vector<std::vector<float>>* fea_values,
                          int fea_dim,
                          const std::vector<std::string>& var_emb_names);

  // Pull sparse variables from server in async mode
  // Param<in>: scope, table_id, var_names, fea_keys, fea_dim
  // Param<out>: fea_values std::future
  std::future<int32_t> PullSparseVarsAsync(
      const Scope& scope, const uint64_t table_id,
      const std::vector<std::string>& var_names,
      std::vector<uint64_t>* fea_keys,
      std::vector<std::vector<float>>* fea_values, int fea_dim);

  // Pull sparse variables from server in sync mode
  // pull immediately to tensors
  // is_training is true means training, false means inference, the behavior is
  // different on pserver

  void PullSparseToTensorSync(const uint64_t table_id, int fea_dim,
                              uint64_t padding_id, platform::Place place,
                              bool is_training,
                              std::vector<const LoDTensor*>* inputs,  // NOLINT
                              std::vector<LoDTensor*>* outputs);      // NOLINT

  // pull dense variables from server in sync mod
  // Param<in>: scope, table_id, var_names
  // Param<out>: void
  void PullDenseVarsSync(const Scope& scope, const uint64_t table_id,
                         const std::vector<std::string>& var_names);

  // pull dense variables from server in async mod
  // Param<in>: scope, table_id, var_names
  // Param<out>: pull_dense_status
  void PullDenseVarsAsync(const Scope& scope, const uint64_t table_id,
                          const std::vector<std::string>& var_names,
                          std::vector<std::future<int32_t>>* pull_dense_status,
                          bool in_cpu);

  // push dense parameters(not gradients) to server in sync mode
  void PushDenseParamSync(const Scope& scope, const uint64_t table_id,
                          const std::vector<std::string>& var_names);

  void PushDenseVarsAsync(const Scope& scope, const uint64_t table_id,
                          const std::vector<std::string>& var_names,
                          std::vector<std::future<int32_t>>* push_sparse_status,
                          float scale_datanorm, int batch_size);

  // push dense variables to server in sync mode
  void PushDenseVarsSync(Scope* scope, const uint64_t table_id,
                         const std::vector<std::string>& var_names);

  void PushSparseVarsAsync(
      const Scope& scope, const uint64_t table_id, const std::string& grad,
      std::vector<std::future<int32_t>>* push_sparse_status);
  // This is specially designed for click/show stats in server
  // Param<in>: scope, table_id, fea_keys, fea_labels, sparse_key_names,
  //            sparse_grad_names, batch_size, use_cvm, dump_slot
  // Param<out>: push_values, push_sparse_status
  void PushSparseVarsWithLabelAsync(
      const Scope& scope, const uint64_t table_id,
      const std::vector<uint64_t>& fea_keys,
      const std::vector<float>& fea_labels,
      const std::vector<std::string>& sparse_key_names,
      const std::vector<std::string>& sparse_grad_names, const int emb_dim,
      std::vector<std::vector<float>>* push_values,
      std::vector<std::future<int32_t>>* push_sparse_status,
      const int batch_size, const bool use_cvm, const bool dump_slot,
      std::vector<uint64_t>* sparse_push_keys, const bool no_cvm);

  // Push sparse variables to server in async mode
  void PushSparseFromTensorWithLabelAsync(
      const Scope& scope, const uint64_t table_id, int fea_dim,
      uint64_t padding_id, bool scale_sparse, const std::string& accesor,
      const std::string& click_name, platform::Place place,
      const std::vector<std::string>& input_names,
      std::vector<const LoDTensor*>* inputs,    // NOLINT
      std::vector<const LoDTensor*>* outputs);  // NOLINT
  void PushSparseFromTensorAsync(const uint64_t table_id, int fea_dim,
                                 uint64_t padding_id, platform::Place place,
                                 std::vector<const LoDTensor*>* inputs,
                                 const LoDTensor* shows,
                                 const LoDTensor* clicks,
                                 std::vector<LoDTensor*>* outputs);
  // Push sparse variables to server in Async mode
  // Param<In>: scope, table_id, fea_keys, sparse_grad_names
  // Param<Out>: push_values, push_sparse_status

  // init server
  void LoadSparseOnServer(const std::string& path, const std::string& meta,
                          uint32_t table_id);
  // init server
  // void InitServer(const std::string& dist_desc,
  //                 const std::vector<uint64_t>& host_sign_list, int index);
  void InitServer(
      const std::string& dist_desc,
      const std::vector<std::string>& host_sign_list, int index, int trainers,
      const std::vector<framework::ProgramDesc>& server_sub_program = {});
  // init trainer
  void InitWorker(const std::string& dist_desc,
                  const std::vector<std::string>& host_sign_list, Scope* scope,
                  const RpcCtxMap& send_ctx,
                  const std::unordered_map<uint64_t, std::vector<std::string>>&
                      dense_varnames,
                  const std::map<std::string, std::string>& envs, int node_num,
                  int index);

  // stop server
  void StopServer();
  // finalize worker to make worker can be stop
  void FinalizeWorker();
  // run server with ip port
  uint64_t RunServer(const std::string& ip, uint32_t port);
  // get client info
  std::vector<uint64_t> GetClientsInfo();
  // create client to client connection
  void CreateClient2ClientConnection();
  // flush all push requests
  void ClientFlush();

  // barrier with barrier table
  void BarrierWithTable(uint32_t barrier_type);

  void PrintTableStat(const uint64_t table_id);
  // mode = 0, load all feature
  // mode = 1, load delta feature, which means load diff
  void LoadModel(const std::string& path, const int mode);
  // mode = 0, load all feature
  // mode = 1, load delta feature, which means load diff
  void LoadModelOneTable(const uint64_t table_id, const std::string& path,
                         const int mode);
  // mode = 0, save all feature
  // mode = 1, save delta feature, which means save diff
  void SaveModel(const std::string& path, const int mode);
  // mode = 0, save all feature
  // mode = 1, save delta feature, which means save diff
  void SaveModelOneTable(const uint64_t table_id, const std::string& path,
                         const int mode);

  // recv table from server and save it in LodTensor
  void RecvAndSaveTable(const uint64_t table_id, const std::string& path);

  // clear all models, release their memory
  void ClearModel();
  // clear one table
  void ClearOneTable(const uint64_t table_id);
  // shrink sparse table
  void ShrinkSparseTable(int table_id, int threshold);
  // shrink dense table
  void ShrinkDenseTable(int table_id, Scope* scope,
                        std::vector<std::string> var_list, float decay,
                        int emb_dim);

  typedef std::function<int32_t(int, int, const std::string&)> MsgHandlerFunc;
  // register client to client communication
  int RegisterClientToClientMsgHandler(int msg_type, MsgHandlerFunc handler);
  // send client to client message
  std::future<int32_t> SendClientToClientMsg(int msg_type, int to_client_id,
                                             const std::string& msg);

  // FleetWrapper singleton
  static std::shared_ptr<FleetWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::distributed::FleetWrapper());
    }
    return s_instance_;
  }
  // this performs better than rand_r, especially large data
  std::default_random_engine& LocalRandomEngine();

  static std::shared_ptr<paddle::distributed::PSCore> pserver_ptr_;

 private:
  static std::shared_ptr<FleetWrapper> s_instance_;
  size_t GetAbsoluteSum(size_t start, size_t end, size_t level,
                        const framework::LoD& lod);

 protected:
  static bool is_initialized_;
  std::map<uint64_t, std::vector<paddle::distributed::Region>> _regions;
  bool scale_sparse_gradient_with_batch_size_;
  int32_t sleep_seconds_before_fail_exit_;
  int client2client_request_timeout_ms_;
  int client2client_connect_timeout_ms_;
  int client2client_max_retry_;
  DISABLE_COPY_AND_ASSIGN(FleetWrapper);
};

}  // end namespace distributed
}  // end namespace paddle
