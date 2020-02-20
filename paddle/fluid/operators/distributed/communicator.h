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

#pragma once

#include <ThreadPool.h>
#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "gflags/gflags.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/rpc_client.h"
#include "paddle/fluid/operators/distributed/rpc_common.h"
#include "paddle/fluid/operators/distributed_ops/send_recv_util.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/split.h"

DECLARE_bool(communicator_is_sgd_optimizer);

namespace paddle {
namespace operators {
namespace distributed {

using Scope = framework::Scope;
using Variable = framework::Variable;

template <typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(size_t capacity) : capacity_(capacity) {
    PADDLE_ENFORCE_GT(capacity_, 0, "The capacity must be greater than 0.");
  }

  bool Push(const T& elem) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return queue_.size() < capacity_; });
      PADDLE_ENFORCE_LT(queue_.size(), capacity_);
      queue_.push_back(elem);
    }
    cv_.notify_one();
    return true;
  }

  bool Push(T&& elem) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return queue_.size() < capacity_; });
      PADDLE_ENFORCE_LT(queue_.size(), capacity_);
      queue_.emplace_back(std::move(elem));
    }
    cv_.notify_one();
    return true;
  }

  T Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [=] { return !queue_.empty(); });
    T rc(std::move(queue_.front()));
    queue_.pop_front();
    cv_.notify_one();
    return rc;
  }

  size_t Cap() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return capacity_;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  const size_t capacity_;
  std::deque<T> queue_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
};

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename T>
inline void MergeVars(const std::string& var_name,
                      const std::vector<std::shared_ptr<Variable>>& vars,
                      Scope* scope, bool merge_add = true) {
  PADDLE_ENFORCE(!vars.empty(), "should have value to merge!");
  auto cpu_place = platform::CPUPlace();
  auto& var0 = vars[0];
  auto* out_var = scope->Var(var_name);
  if (var0->IsType<framework::LoDTensor>()) {
    auto dims = var0->Get<framework::LoDTensor>().dims();
    VLOG(3) << "merge " << var_name << " LoDTensor dims " << dims
            << "; merge add: " << merge_add;
    // init output tensor
    auto* out_t = out_var->GetMutable<framework::LoDTensor>();
    out_t->mutable_data<T>(dims, cpu_place);
    // check the input dims
    for (auto& var : vars) {
      auto& var_t = var->Get<framework::LoDTensor>();
      PADDLE_ENFORCE_EQ(var_t.dims(), dims, "should have the same dims");
    }

    // set output tensor to 0.
    auto cpu_ctx = paddle::platform::CPUDeviceContext();
    math::SetConstant<paddle::platform::CPUDeviceContext, T> constant_functor;
    constant_functor(cpu_ctx, out_t, static_cast<T>(0));
    // sum all vars to out
    auto result = EigenVector<T>::Flatten(*out_t);
    for (auto& var : vars) {
      auto& in_t = var->Get<framework::LoDTensor>();
      auto in = EigenVector<T>::Flatten(in_t);
      result.device(*cpu_ctx.eigen_device()) = result + in;
    }
    if (!merge_add) {
      result.device(*cpu_ctx.eigen_device()) =
          result / static_cast<T>(vars.size());
    }
  } else if (var0->IsType<framework::SelectedRows>()) {
    auto& slr0 = var0->Get<framework::SelectedRows>();
    auto* out_slr = out_var->GetMutable<framework::SelectedRows>();
    out_slr->mutable_rows()->clear();
    out_slr->mutable_value()->mutable_data<T>({{}}, cpu_place);
    std::vector<const paddle::framework::SelectedRows*> inputs;
    inputs.reserve(vars.size());
    for (auto& var : vars) {
      inputs.push_back(&var->Get<framework::SelectedRows>());
    }
    auto dev_ctx = paddle::platform::CPUDeviceContext();
    if (merge_add) {
      math::scatter::MergeAdd<paddle::platform::CPUDeviceContext, T> merge_add;
      merge_add(dev_ctx, inputs, out_slr);
    } else {
      math::scatter::MergeAverage<paddle::platform::CPUDeviceContext, T>
          merge_average;
      merge_average(dev_ctx, inputs, out_slr);
    }

    VLOG(3) << "merge " << var_name << " SelectedRows height: " << slr0.height()
            << " dims: " << slr0.value().dims() << "; merge add: " << merge_add;
  } else {
    PADDLE_THROW("unsupported var type!");
  }
}

using RpcCtxMap = std::unordered_map<std::string, RpcContext>;

class Communicator {
 public:
  Communicator();
  explicit Communicator(const std::map<std::string, std::string>& envs);
  virtual ~Communicator() {}

  virtual void Start() = 0;
  virtual void Stop() = 0;
  virtual bool IsRunning() { return running_; }

  virtual void Send(const std::vector<std::string>& var_names,
                    const std::vector<std::string>& var_tables,
                    const framework::Scope& scope) = 0;

  virtual void Recv() = 0;

  virtual void Barrier() {}
  virtual void BarrierTriggerDecrement() {}
  virtual void BarrierTriggerReset(int init_counter) {}

  virtual void InitImpl(const RpcCtxMap& send_varname_to_ctx,
                        const RpcCtxMap& recv_varname_to_ctx,
                        Scope* recv_scope) {}
  virtual void InitImpl(const paddle::framework::ProgramDesc& program,
                        Scope* recv_scope) = 0;

  static Communicator* GetInstance() { return communicator_.get(); }
  static std::shared_ptr<Communicator> GetInstantcePtr() {
    return communicator_;
  }
  template <typename T>
  static Communicator* InitInstance(
      const paddle::framework::ProgramDesc& program, Scope* recv_scope,
      const std::map<std::string, std::string>& envs) {
    std::call_once(init_flag_, &Communicator::InitWithProgram<T>, program,
                   recv_scope, std::ref(envs));
    return communicator_.get();
  }

  template <typename T>
  static void InitWithProgram(const paddle::framework::ProgramDesc& program,
                              Scope* recv_scope,
                              const std::map<std::string, std::string>& envs) {
    if (communicator_.get() == nullptr) {
      communicator_.reset(new T(std::ref(envs)));
      communicator_->InitImpl(program, recv_scope);
    }
  }

 protected:
  bool running_ = false;
  static std::shared_ptr<Communicator> communicator_;
  static std::once_flag init_flag_;
  std::unordered_map<std::string, std::string> envs;
};

using SparseIdsMap =
    std::unordered_map<std::string, std::vector<std::unordered_set<int64_t>>>;

class AsyncCommunicator : public Communicator {
 public:
  AsyncCommunicator() : Communicator() {}
  explicit AsyncCommunicator(const std::map<std::string, std::string>& envs)
      : Communicator(envs) {
    independent_recv_thread_ = static_cast<bool>(
        std::stoi(envs.at("communicator_independent_recv_thread")));
    min_send_grad_num_before_recv_ =
        std::stoi(envs.at("communicator_min_send_grad_num_before_recv"));
    thread_pool_size_ = std::stoi(envs.at("communicator_thread_pool_size"));
    max_merge_var_num_ = std::stoi(envs.at("communicator_max_merge_var_num"));
    send_wait_times_ = std::stoi(envs.at("communicator_send_wait_times"));
    send_queue_size_ = std::stoi(envs.at("communicator_send_queue_size"));
    is_sgd_optimizer_ =
        static_cast<bool>(std::stoi(envs.at("communicator_is_sgd_optimizer")));
    VLOG(0) << "AsyncCommunicator Initialized";
  }
  ~AsyncCommunicator();
  void Start() override;
  void Stop() override;

  void Recv() override;
  void RecvAll();

  void InitImpl(const RpcCtxMap& send_varname_to_ctx,
                const RpcCtxMap& recv_varname_to_ctx,
                Scope* recv_scope) override;

  void InitImpl(const paddle::framework::ProgramDesc& program,
                Scope* recv_scope) override;

  void SendThread();
  void RecvThread();

  void Send(const std::vector<std::string>& var_names,
            const std::vector<std::string>& var_tables,
            const framework::Scope& scope) override;

 private:
  int min_send_grad_num_before_recv_;
  int thread_pool_size_;
  int max_merge_var_num_;
  int send_wait_times_;
  int send_queue_size_;
  bool independent_recv_thread_;
  bool is_sgd_optimizer_;

 private:
  std::unordered_map<std::string,
                     std::shared_ptr<BlockingQueue<std::shared_ptr<Variable>>>>
      send_varname_to_queue_;
  RpcCtxMap send_varname_to_ctx_;
  RpcCtxMap recv_varname_to_ctx_;
  std::unique_ptr<std::thread> send_thread_{nullptr};
  std::unique_ptr<std::thread> recv_thread_{nullptr};
  Scope* recv_scope_;                  // should be global scope
  std::unique_ptr<Scope> send_scope_;  // an independent scope
  std::unique_ptr<::ThreadPool> send_threadpool_{nullptr};
  std::unique_ptr<::ThreadPool> recv_threadpool_{nullptr};
  std::atomic_uint grad_num_{0};  // the num of gradient sent since last recv
};

class HalfAsyncCommunicator : public Communicator {
 public:
  HalfAsyncCommunicator() {}
  explicit HalfAsyncCommunicator(const std::map<std::string, std::string>& envs)
      : Communicator(envs) {
    max_merge_var_num_ = std::stoi(envs.at("communicator_max_merge_var_num"));
    send_wait_times_ = std::stoi(envs.at("communicator_send_wait_times"));
    thread_pool_size_ = std::stoi(envs.at("communicator_thread_pool_size"));
    send_queue_size_ = std::stoi(envs.at("communicator_send_queue_size"));
    VLOG(0) << "HalfAsyncCommunicator Initialized";
  }
  ~HalfAsyncCommunicator();
  void Start() override;
  void Stop() override;

  void Send(const std::vector<std::string>& var_names,
            const std::vector<std::string>& var_tables,
            const framework::Scope& scope) override;

  void Recv() override;

  void Barrier() override;
  void BarrierWeakUp();

  void BarrierTriggerDecrement() override;
  void BarrierTriggerReset(int initial_val) override;

  void InitImpl(const RpcCtxMap& send_varname_to_ctx,
                const RpcCtxMap& recv_varname_to_ctx,
                Scope* recv_scope) override;

  void InitImpl(const paddle::framework::ProgramDesc& program,
                Scope* recv_scope) override;

  void ConsumeThread();
  virtual void BarrierSend() {}
  virtual void BarrierRecv() {}

 protected:
  int max_merge_var_num_;
  int send_wait_times_;
  int thread_pool_size_;
  int send_queue_size_;
  int trainer_id_ = 0;

 protected:
  std::unordered_map<std::string,
                     std::shared_ptr<BlockingQueue<std::shared_ptr<Variable>>>>
      send_varname_to_queue_;
  RpcCtxMap send_varname_to_ctx_;
  RpcCtxMap recv_varname_to_ctx_;
  std::unique_ptr<std::thread> consume_thread_{nullptr};
  Scope* recv_scope_;                  // should be global scope
  std::unique_ptr<Scope> send_scope_;  // an independent scope
  std::unique_ptr<::ThreadPool> consume_threadpool_{nullptr};
  std::unique_ptr<::ThreadPool> recv_threadpool_{nullptr};

  // mutex for Wait for barrier
  std::mutex barrier_mutex_;
  std::condition_variable barrier_cond_;
  std::atomic<int64_t> barrier_trigger_{0};
  std::atomic<int64_t> barrier_counter_{0};
};

class SyncCommunicator : public HalfAsyncCommunicator {
 public:
  SyncCommunicator() : HalfAsyncCommunicator() {}
  explicit SyncCommunicator(const std::map<std::string, std::string>& envs)
      : HalfAsyncCommunicator(envs) {
    trainer_id_ = std::stoi(envs.at("trainer_id"));
    auto pserver_strings = envs.at("pserver_endpoints");
    pserver_endpoints_ = paddle::string::Split(pserver_strings, ',');
    VLOG(0) << "SyncCommunicator Initialized";
  }
  ~SyncCommunicator();
  void BarrierSend();
  void BarrierRecv();

 private:
  std::vector<std::string> pserver_endpoints_{};
};

class GeoSgdCommunicator : public Communicator {
 public:
  GeoSgdCommunicator() : Communicator() {}
  explicit GeoSgdCommunicator(const std::map<std::string, std::string>& envs)
      : Communicator(envs) {
    geo_need_push_nums_ = std::stoi(envs.at("geo_need_push_nums"));
    trainer_nums_ = std::stoi(envs.at("geo_trainer_nums"));
    thread_pool_size_ = std::stoi(envs.at("communicator_thread_pool_size"));
    send_wait_times_ = std::stoi(envs.at("communicator_send_wait_times"));
    VLOG(0) << "GeoSgdCommunicator Initialized";
  }

  ~GeoSgdCommunicator();

  void Start() override;
  void Stop() override;

  void Send(const std::vector<std::string>& var_names,
            const std::vector<std::string>& var_tables,
            const framework::Scope& scope) override;

  void Recv() override;

  void InitImpl(const paddle::framework::ProgramDesc& program,
                Scope* recv_scope) override;

 private:
  void SendThread();
  std::unordered_set<int64_t> SparseIdsMerge(
      const std::vector<SparseIdsMap>& ids_send_vec,
      const std::string& var_name, const std::string& splited_var_name);

  void SendUpdateDenseVars(const std::string& var_name,
                           const std::string& splited_var_name);

  void SendUpdateSparseVars(const std::string& var_name,
                            const std::string& splited_var_name,
                            const std::unordered_set<int64_t>& ids_table);

  void RecvUpdateDenseVars(const std::string& var_name,
                           const std::string& splited_var_name);
  void RecvUpdateSparseVars(const std::string& var_name,
                            const std::string& splited_var_name);

  void GeoSgdDenseParamInit(framework::Scope* scope_x,
                            framework::Scope* scope_y,
                            const std::string var_name);

  void GeoSgdSparseParamInit(framework::Scope* scope_x,
                             framework::Scope* scope_y,
                             const std::string var_name);

  void RpcSend(const std::string& origin_var_name,
               const std::string& splited_var_name,
               const size_t& splited_var_index);

  void RpcRecv(const std::string& origin_var_name,
               const std::string& splited_var_name,
               const size_t& splited_var_index);

  const std::string VarToDeltaVar(const std::string var_name) {
    std::string delta_name = var_name;
    const std::string send_name = delta_name.append(".delta");
    return send_name;
  }

  const std::string DeltaVarToVar(const std::string var_name) {
    std::string origin_name = var_name;
    origin_name.erase(origin_name.find(".delta"), 6);
    const std::string param_name = origin_name;
    return param_name;
  }

  size_t GetSplitedVarIndex(const std::string var_name,
                            const std::string splited_var_name) {
    size_t index = 0;
    for (size_t i = 0;
         i < send_varname_to_ctx_[var_name].splited_var_names.size(); i++) {
      if (send_varname_to_ctx_[var_name].splited_var_names[i] ==
          splited_var_name) {
        index = i;
        break;
      }
    }
    return index;
  }

 private:
  int trainer_nums_ = 1;
  int geo_need_push_nums_ = 100;
  int thread_pool_size_;
  int send_wait_times_;

 private:
  int send_var_nums_ = 0;

  RpcCtxMap send_varname_to_ctx_;
  RpcCtxMap recv_varname_to_ctx_;

  // parameter for local training
  Scope* training_scope_;

  // parameter for delta calc and send
  std::shared_ptr<Scope> delta_scope_;

  // parameter for storage the pserver param after last recv
  std::shared_ptr<Scope> old_scope_;

  // parameter on pserver
  std::shared_ptr<Scope> pserver_scope_;

  // if var is sparse, using selected rows, bool=true
  std::unordered_map<std::string, bool> var_list_;

  std::shared_ptr<BlockingQueue<std::shared_ptr<SparseIdsMap>>>
      need_push_queue_;
  std::vector<SparseIdsMap> ids_send_vec_;

  std::unordered_map<std::string, std::vector<int64_t>> absolute_section_;
  std::unordered_map<std::string, int64_t> vars_first_dimension_;

  std::unique_ptr<::ThreadPool> send_threadpool_{nullptr};
  std::unique_ptr<std::thread> send_thread_{nullptr};

  size_t need_thread_nums_{0};
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
