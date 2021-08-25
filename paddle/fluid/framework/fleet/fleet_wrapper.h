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
#include <ThreadPool.h>
#include <atomic>
#include <ctime>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/fleet/context/trainer_context.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
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
    pull_local_thread_num_ = 25;
    trainer_context_ = std::make_shared<FeedTrainerContext>();
  }

  // set client to client communication config
  void SetClient2ClientConfig(int request_timeout_ms, int connect_timeout_ms,
                              int max_retry);

  void SetPullLocalThreadNum(int thread_num) {
    pull_local_thread_num_ = thread_num;
  }

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
  void PullSparseToTensorSync(const uint64_t table_id, int fea_dim,
                              uint64_t padding_id, platform::Place place,
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
  void PullDenseVarsAsync(
      const Scope& scope, const uint64_t table_id,
      const std::vector<std::string>& var_names,
      std::vector<::std::future<int32_t>>* pull_dense_status);

  // push dense parameters(not gradients) to server in sync mode
  void PushDenseParamSync(const Scope& scope, const uint64_t table_id,
                          const std::vector<std::string>& var_names);

  // Push dense variables to server in async mode
  // Param<in>: scope, table_id, var_names, scale_datanorm, batch_size
  // Param<out>: push_sparse_status
  void PushDenseVarsAsync(
      const Scope& scope, const uint64_t table_id,
      const std::vector<std::string>& var_names,
      std::vector<::std::future<int32_t>>* push_sparse_status,
      float scale_datanorm, int batch_size);

  // push dense variables to server in sync mode
  void PushDenseVarsSync(Scope* scope, const uint64_t table_id,
                         const std::vector<std::string>& var_names);

  // Push sparse variables with labels to server in async mode
  std::vector<std::unordered_map<uint64_t, std::vector<float>>> local_tables_;
  void PullSparseToLocal(const uint64_t table_id, int fea_value_dim);
  void PullSparseVarsFromLocal(const Scope& scope, const uint64_t table_id,
                               const std::vector<std::string>& var_names,
                               std::vector<uint64_t>* fea_keys,
                               std::vector<std::vector<float>>* fea_values,
                               int fea_value_dim);
  void ClearLocalTable();
  std::vector<std::unordered_map<uint64_t, std::vector<float>>>&
  GetLocalTable() {
    return local_tables_;
  }

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
      std::vector<::std::future<int32_t>>* push_sparse_status,
      const int batch_size, const bool use_cvm, const bool dump_slot,
      std::vector<uint64_t>* sparse_push_keys, const bool no_cvm,
      const bool scale_sparse_gradient_with_batch_size);

  // Push sparse variables to server in async mode
  void PushSparseFromTensorWithLabelAsync(
      const Scope& scope, const uint64_t table_id, int fea_dim,
      uint64_t padding_id, bool scale_sparse, const std::string& accesor,
      const std::string& click_name, platform::Place place,
      const std::vector<std::string>& input_names,
      std::vector<const LoDTensor*>* inputs,    // NOLINT
      std::vector<const LoDTensor*>* outputs);  // NOLINT

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

  // init server
  void InitServer(const std::string& dist_desc, int index, size_t role_comm);
  // init trainer
  void InitWorker(const std::string& dist_desc,
                  const std::vector<uint64_t>& host_sign_list, int node_num,
                  int index, size_t role_comm);
  // stop server
  void StopServer();
  // finalize worker to make worker can be stop
  void FinalizeWorker();
  // run server
  uint64_t RunServer();
  // run server with ip port
  uint64_t RunServer(const std::string& ip, uint32_t port);
  // gather server ip
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
  void SaveMultiTableOnePath(const std::vector<int>& table_ids,
                             const std::string& path, const int mode);
  // mode = 0, save all feature
  // mode = 1, save delta feature, which means save diff
  void SaveModelOneTable(const uint64_t table_id, const std::string& path,
                         const int mode);
  // save model with prefix
  void SaveModelOneTablePrefix(const uint64_t table_id, const std::string& path,
                               const int mode, const std::string& prefix);
  // get save cache threshold
  double GetCacheThreshold(int table_id);
  // shuffle cache model between servers
  void CacheShuffle(int table_id, const std::string& path, const int mode,
                    const double cache_threshold);
  // save cache model
  // cache model can speed up online predict
  int32_t SaveCache(int table_id, const std::string& path, const int mode);
  // save sparse table filtered by user-defined whitelist
  int32_t SaveWithWhitelist(int table_id, const std::string& path,
                            const int mode, const std::string& whitelist_path);
  void LoadWithWhitelist(const uint64_t table_id, const std::string& path,
                         const int mode);
  // copy feasign key/value from src_table_id to dest_table_id
  int32_t CopyTable(const uint64_t src_table_id, const uint64_t dest_table_id);
  // copy feasign key/value from src_table_id to dest_table_id
  int32_t CopyTableByFeasign(const uint64_t src_table_id,
                             const uint64_t dest_table_id,
                             const std::vector<uint64_t>& feasign_list);
  // clear all models, release their memory
  void ClearModel();
  // clear one table
  void ClearOneTable(const uint64_t table_id);
  // shrink sparse table
  void ShrinkSparseTable(int table_id);
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
  // confirm all the updated params in the current pass
  void Confirm();
  // revert all the updated params in the current pass
  void Revert();
  // FleetWrapper singleton
  static std::shared_ptr<FleetWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::FleetWrapper());
    }
    return s_instance_;
  }
  // this performs better than rand_r, especially large data
  std::default_random_engine& LocalRandomEngine();
  void InitTrainerScope(Scope* scope);

#ifdef PADDLE_WITH_PSLIB
  static std::shared_ptr<paddle::distributed::PSlib> pslib_ptr_;
#endif

 private:
  static std::shared_ptr<FleetWrapper> s_instance_;
#ifdef PADDLE_WITH_PSLIB
  std::map<uint64_t, std::vector<paddle::ps::Region>> _regions;
#endif

  size_t GetAbsoluteSum(size_t start, size_t end, size_t level,
                        const framework::LoD& lod);

 protected:
  static bool is_initialized_;
  bool scale_sparse_gradient_with_batch_size_;
  int32_t sleep_seconds_before_fail_exit_;
  int client2client_request_timeout_ms_;
  int client2client_connect_timeout_ms_;
  int client2client_max_retry_;
  std::unique_ptr<::ThreadPool> local_pull_pool_{nullptr};
  int pull_local_thread_num_;
  std::unique_ptr<::ThreadPool> pull_to_local_pool_{nullptr};
  int local_table_shard_num_;
  std::shared_ptr<FeedTrainerContext> trainer_context_;
  DISABLE_COPY_AND_ASSIGN(FleetWrapper);
};

template <typename T>
paddle::framework::Channel<T> GlobalShuffle(
    TrainerContextInterface* context, paddle::framework::Channel<T> in_chan,
    decltype(std::function<size_t(const T&)>()) func, int thread_num = 12) {
  platform::Timer timeline;
  timeline.Start();

  auto* trainer_context = (FeedTrainerContext*)context;  // NOLINT
  auto comm = trainer_context->comm_;
  trainer_context->MPI()->Barrier(comm);
  size_t worker_num = trainer_context->MPI()->Size(comm);
  paddle::framework::Channel<T> out_chan = paddle::framework::MakeChannel<T>();
  auto fleet_ptr = paddle::framework::FleetWrapper::GetInstance();
  fleet_ptr->RegisterClientToClientMsgHandler(
      0,
      [out_chan](int msg_type, int client_id, const std::string& msg) -> int {
        VLOG(3) << "Receive msg_type=" << msg_type
                << ", client_id=" << client_id
                << ", msg length=" << msg.length();
        if (msg.length() == 0) {
          return 0;
        }
        paddle::framework::BinaryArchive ar;
        ar.SetReadBuffer(const_cast<char*>(msg.c_str()), msg.length(), nullptr);
        if (ar.Cursor() == ar.Finish()) {
          return 0;
        }
        std::vector<T> data;
        while (ar.Cursor() < ar.Finish()) {
          data.push_back(ar.Get<T>());
        }
        CHECK(ar.Cursor() == ar.Finish());
        out_chan->Write(std::move(data));
        data.clear();
        return 0;
      });
  in_chan->SetBlockSize(512);
  VLOG(3) << "Shuffle() data_chan size " << in_chan->Size();
  auto get_client_id = [worker_num, func](const T& data) -> size_t {
    return func(data) % worker_num;
  };
  auto global_shuffle_func = [in_chan, worker_num, get_client_id]() {
    auto fleet_ptr = paddle::framework::FleetWrapper::GetInstance();
    std::vector<T> data;
    while (in_chan->Read(data)) {
      std::vector<paddle::framework::BinaryArchive> ars(worker_num);
      for (auto& d : data) {
        auto client_id = get_client_id(d);
        ars[client_id] << d;
      }
      std::vector<std::future<int32_t>> total_status;
      for (size_t i = 0; i < worker_num; ++i) {
        if (ars[i].Length() == 0) {
          continue;
        }
        std::string msg(ars[i].Buffer(), ars[i].Length());
        auto ret = fleet_ptr->SendClientToClientMsg(0, i, msg);
        total_status.push_back(std::move(ret));
      }
      for (auto& t : total_status) {
        t.wait();
      }
      ars.clear();
      data.clear();
    }
  };
  trainer_context->MPI()->Barrier(comm);
  std::vector<std::thread> global_shuffle_threads;
  for (int i = 0; i < thread_num; ++i) {
    global_shuffle_threads.push_back(std::thread(global_shuffle_func));
  }
  for (std::thread& t : global_shuffle_threads) {
    t.join();
  }
  global_shuffle_threads.clear();
  in_chan->Clear();
  trainer_context->MPI()->Barrier(comm);
  out_chan->Close();
  timeline.Pause();
  VLOG(0) << "Shuffle() end, cost time=" << timeline.ElapsedSec() << " seconds";
  return out_chan;
}

}  // end namespace framework
}  // end namespace paddle
