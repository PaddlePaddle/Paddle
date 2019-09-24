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

#pragma once

#include <memory>
#ifdef PADDLE_WITH_PSLIB
#include <archive.h>
#include <pslib.h>
#endif
#include <atomic>
#include <ctime>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace framework {

// A wrapper class for pslib.h, this class follows Singleton pattern
// i.e. only initialized once in the current process
// Example:
//    std::shared_ptr<FleetWrapper> fleet_ptr =
//         FleetWrapper::GetInstance();
//    string dist_desc;
//    fleet_ptr->InitServer(dist_desc, 0);
// interface design principles:
// Pull
//   Sync: PullSparseVarsSync
//   Async: PullSparseVarsAsync(not implemented currently)
// Push
//   Sync: PushSparseVarsSync
//   Async: PushSparseVarsAsync(not implemented currently)
//   Async: PushSparseVarsWithLabelAsync(with special usage)
// Push dense variables to server in Async mode
// Param<in>: scope, table_id, var_names
// Param<out>: push_sparse_status

class FleetWrapper {
 public:
  virtual ~FleetWrapper() {}
  FleetWrapper() {
    scale_sparse_gradient_with_batch_size_ = true;
    // trainer sleep some time for pslib core dump
    sleep_seconds_before_fail_exit_ = 300;
    // pslib request server timeout ms
    client2client_request_timeout_ms_ = 500000;
    // pslib connect server timeout_ms
    client2client_connect_timeout_ms_ = 10000;
    // pslib request max retry
    client2client_max_retry_ = 3;
  }

  void SetClient2ClientConfig(int request_timeout_ms, int connect_timeout_ms,
                              int max_retry);

  // Pull sparse variables from server in Sync mode
  // Param<in>: scope, table_id, var_names, fea_keys
  // Param<out>: fea_values
  void PullSparseVarsSync(const Scope& scope, const uint64_t table_id,
                          const std::vector<std::string>& var_names,
                          std::vector<uint64_t>* fea_keys,
                          std::vector<std::vector<float>>* fea_values,
                          int fea_dim);

  void PullDenseVarsSync(const Scope& scope, const uint64_t table_id,
                         const std::vector<std::string>& var_names);

  void PullDenseVarsAsync(
      const Scope& scope, const uint64_t table_id,
      const std::vector<std::string>& var_names,
      std::vector<::std::future<int32_t>>* pull_dense_status);

  void PushDenseParamSync(const Scope& scope, const uint64_t table_id,
                          const std::vector<std::string>& var_names);

  // Push dense variables to server in async mode
  // Param<in>: scope, table_id, var_names,
  // Param<out>: push_sparse_status
  void PushDenseVarsAsync(
      const Scope& scope, const uint64_t table_id,
      const std::vector<std::string>& var_names,
      std::vector<::std::future<int32_t>>* push_sparse_status,
      float scale_datanorm, int batch_size);

  void PushDenseVarsSync(Scope* scope, const uint64_t table_id,
                         const std::vector<std::string>& var_names);

  // Push sparse variables with labels to server in Async mode
  // This is specially designed for click/show stats in server
  // Param<in>: scope, table_id, var_grad_names,
  //            fea_keys, fea_labels, sparse_grad_names
  // Param<out>: push_values, push_sparse_status
  void PushSparseVarsWithLabelAsync(
      const Scope& scope, const uint64_t table_id,
      const std::vector<uint64_t>& fea_keys,
      const std::vector<float>& fea_labels,
      const std::vector<std::string>& sparse_key_names,
      const std::vector<std::string>& sparse_grad_names, const int emb_dim,
      std::vector<std::vector<float>>* push_values,
      std::vector<::std::future<int32_t>>* push_sparse_status,
      const int batch_size, const bool use_cvm, const bool dump_slot);

  // Push sparse variables to server in Async mode
  // Param<In>: scope, table_id, fea_keys, sparse_grad_names
  // Param<Out>: push_values, push_sparse_status
  /*
  void PushSparseVarsAsync(
          const Scope& scope,
          const uint64_t table_id,
          const std::vector<uint64_t>& fea_keys,
          const std::vector<std::string>& sparse_grad_names,
          std::vector<std::vector<float>>* push_values,
          std::vector<::std::future<int32_t>>* push_sparse_status);
  */

  void InitServer(const std::string& dist_desc, int index);
  void InitWorker(const std::string& dist_desc,
                  const std::vector<uint64_t>& host_sign_list, int node_num,
                  int index);
  void StopServer();
  uint64_t RunServer();
  void GatherServers(const std::vector<uint64_t>& host_sign_list, int node_num);
  // gather client ip
  void GatherClients(const std::vector<uint64_t>& host_sign_list);
  // get client info
  std::vector<uint64_t> GetClientsInfo();
  // create client to client connection
  void CreateClient2ClientConnection();

  // flush all push requests
  void ClientFlush();
  // load from paddle model
  void LoadFromPaddleModel(Scope& scope, const uint64_t table_id,  // NOLINT
                           std::vector<std::string> var_list,
                           std::string model_path, std::string model_proto_file,
                           std::vector<std::string> table_var_list,
                           bool load_combine);
  // mode = 0, load all feature
  // mode = 1, laod delta feature, which means load diff
  void LoadModel(const std::string& path, const int mode);
  // mode = 0, load all feature
  // mode = 1, laod delta feature, which means load diff
  void LoadModelOneTable(const uint64_t table_id, const std::string& path,
                         const int mode);
  // mode = 0, save all feature
  // mode = 1, save delta feature, which means save diff
  void SaveModel(const std::string& path, const int mode);

  double GetCacheThreshold();
  void CacheShuffle(int table_id, const std::string& path, const int mode,
                    const double cache_threshold);
  int32_t SaveCache(int table_id, const std::string& path, const int mode);

  void ClearModel();

  void ShrinkSparseTable(int table_id);
  void ShrinkDenseTable(int table_id, Scope* scope,
                        std::vector<std::string> var_list, float decay,
                        int emb_dim);

  // register client to client communication
  typedef std::function<int32_t(int, int, const std::string&)> MsgHandlerFunc;
  int RegisterClientToClientMsgHandler(int msg_type, MsgHandlerFunc handler);
  // send client to client message
  std::future<int32_t> SendClientToClientMsg(int msg_type, int to_client_id,
                                             const std::string& msg);

  template <typename T>
  void Serialize(const std::vector<T*>& t, std::string* str);
  template <typename T>
  void Deserialize(std::vector<T>* t, const std::string& str);
  static std::shared_ptr<FleetWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::FleetWrapper());
    }
    return s_instance_;
  }

  // this performs better than rand_r, especially large data
  std::default_random_engine& LocalRandomEngine();

#ifdef PADDLE_WITH_PSLIB
  static std::shared_ptr<paddle::distributed::PSlib> pslib_ptr_;
#endif

 private:
  static std::shared_ptr<FleetWrapper> s_instance_;
#ifdef PADDLE_WITH_PSLIB
  std::map<uint64_t, std::vector<paddle::ps::Region>> _regions;
#endif

 protected:
  static bool is_initialized_;
  bool scale_sparse_gradient_with_batch_size_;
  int32_t sleep_seconds_before_fail_exit_;
  int client2client_request_timeout_ms_;
  int client2client_connect_timeout_ms_;
  int client2client_max_retry_;
  DISABLE_COPY_AND_ASSIGN(FleetWrapper);
};

}  // end namespace framework
}  // end namespace paddle
