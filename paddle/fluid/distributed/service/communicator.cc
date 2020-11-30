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

#include "paddle/fluid/distributed/service/communicator.h"
#include <google/protobuf/text_format.h>
#include "paddle/fluid/distributed/table/table.h"

#include <gflags/gflags.h>
#include <paddle/fluid/framework/program_desc.h>

#include <algorithm>
#include <chrono>  // NOLINT
#include <map>
#include <thread>  // NOLINT
#include <unordered_set>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/split.h"

namespace paddle {
namespace distributed {

using framework::LoDTensor;
using framework::SelectedRows;

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

Communicator::Communicator() {}

void Communicator::init_gflag(const std::string &gflags) {
  VLOG(0) << "Pslib Init With Gflags:" << gflags;
  std::vector<std::string> flags = paddle::string::split_string(gflags);
  if (flags.size() < 1) {  //不存在则填默认值，向前兼容hard-code的默认配置
    flags.push_back("-max_body_size=314217728");
    flags.push_back("-bthread_concurrency=40");
    flags.push_back("-socket_max_unwritten_bytes=2048000000");
    flags.push_back("-max_connection_pool_size=1950");
  }
  auto it = flags.begin();
  flags.insert(it, "exe default");
  char *flags_ptr[flags.size()];
  for (size_t i = 0; i < flags.size(); ++i) {
    flags_ptr[i] = (char *)(flags[i].c_str());
  }
  int params_cnt = flags.size();
  char **params_ptr = &(flags_ptr[0]);
  ::google::ParseCommandLineFlags(&params_cnt, &params_ptr, true);
}

std::once_flag Communicator::init_flag_;
std::shared_ptr<Communicator> Communicator::communicator_(nullptr);

void AsyncCommunicator::RecvByCommunicator() {
  if (!running_) return;
  RecvNoBarrier();
  VLOG(3) << "run recv graph end";
}

void AsyncCommunicator::RecvNoBarrier() {
  std::vector<std::future<int32_t>> dense_status;
  for (auto &dense_region : _dense_pull_regions) {
    auto tid = dense_region.first;
    auto &regions = dense_region.second;
    auto status = _worker_ptr->pull_dense(regions.data(), regions.size(), tid);
    dense_status.push_back(std::move(status));
  }

  for (int i = 0; i < dense_status.size(); ++i) {
    dense_status[i].wait();
  }

  for (auto &iter : recv_varname_to_ctx_) {
    auto var_names = iter.second;
    for (auto &t : var_names) {
      Variable *var = recv_scope_->FindVar(t);
      LoDTensor *tensor = var->GetMutable<LoDTensor>();
      VLOG(1) << "AsyncCommunicator::RecvNoBarrier Var " << t << " On gpu? "
              << platform::is_gpu_place(tensor->place());
      if (platform::is_gpu_place(tensor->place())) {
#ifdef PADDLE_WITH_CUDA
        LoDTensor *temp_tensor =
            xpu_temp_scope_->FindVar(t)->GetMutable<LoDTensor>();
        framework::TensorCopy(*temp_tensor, tensor->place(), tensor);
#endif
      }
    }
  }

  return;
}

void AsyncCommunicator::SendByCommunicator() {
  for (auto &iter : send_varname_to_ctx_) {
    auto &ctx = iter.second;
    auto &varnames = ctx.origin_varnames;
    auto &table_id = ctx.table_id;
    size_t var_nums = varnames.size();

    auto &check_queue = send_varname_to_queue_[varnames[0]];
    std::vector<std::vector<std::shared_ptr<Variable>>> vars;
    vars.resize(var_nums);
    int merged_var_num = 0;
    int wait_times = 0;
    while (merged_var_num < max_merge_var_num_) {
      if (check_queue->Size() == 0) {
        VLOG(4) << "wait_times -> " << wait_times;
        if (wait_times >= send_wait_times_) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_times++;
        continue;
      } else {
        wait_times = 0;
        for (size_t i = 0; i < var_nums; i++) {
          auto &var_name = varnames[i];
          auto &var_queue = send_varname_to_queue_[var_name];
          vars[i].push_back(var_queue->Pop());
        }
        merged_var_num++;
      }
    }
    if (merged_var_num == 0) continue;

    for (size_t i = 0; i < var_nums; i++) {
      auto &var_name = varnames[i];
      MergeVars<float>(var_name, vars[i], send_scope_.get(), 1);
    }

    if (ctx.is_sparse) {
      PADDLE_ENFORCE_EQ(varnames.size(), 1, "");
      SendSparse(varnames[0], table_id);
    } else {
      // grad_num_.fetch_add(merged_var_num, std::memory_order_relaxed);
      SendDense(ctx);
    }
  }
}

void AsyncCommunicator::MainThread() {
  VLOG(3) << "MainThread start and wait";

  while (waiting_ && running_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    VLOG(3) << "wait for running";
  }

  while (running_) {
    SendByCommunicator();
    BarrierSend();
    RecvByCommunicator();
    BarrierRecv();
    BarrierWeakUp();
  }
  VLOG(1) << "communicator stopped, send thread exit";
}

void AsyncCommunicator::InitImpl(const RpcCtxMap &send_varname_to_ctx,
                                 const RecvCtxMap &recv_varname_to_ctx,
                                 const std::string &dist_desc,
                                 const std::vector<uint64_t> &host_sign_list,
                                 Scope *recv_scope) {
  google::protobuf::TextFormat::ParseFromString(dist_desc, &_ps_param);
  send_varname_to_ctx_ = std::move(send_varname_to_ctx);
  recv_varname_to_ctx_ = std::move(recv_varname_to_ctx);
  recv_scope_ = std::move(recv_scope);
  send_scope_.reset(new Scope());
  xpu_temp_scope_.reset(new Scope());
  for (auto &iter : send_varname_to_ctx_) {
    auto &ctx = iter.second;
    auto &varnames = ctx.origin_varnames;
    for (auto &var_name : varnames) {
      send_varname_to_queue_[var_name] =
          std::make_shared<BlockingQueue<std::shared_ptr<Variable>>>(
              send_queue_size_);
    }
  }

  // not used, just for psclient's init
  for (auto &iter : recv_varname_to_ctx_) {
    auto tid = iter.first;
    auto var_names = iter.second;

    auto &regions = _dense_pull_regions[tid];
    regions.reserve(var_names.size());
    for (auto &t : var_names) {
      Variable *var = recv_scope_->FindVar(t);
      LoDTensor *tensor = var->GetMutable<LoDTensor>();
      VLOG(1) << "AsyncCommunicator::InitImpl Var " << t << " On gpu? "
              << platform::is_gpu_place(tensor->place());
      if (platform::is_gpu_place(tensor->place())) {
#ifdef PADDLE_WITH_CUDA
        Variable *temp_var = xpu_temp_scope_->Var(t);
        LoDTensor *temp_tensor = temp_var->GetMutable<LoDTensor>();
        temp_tensor->Resize(tensor->dims());
        float *temp_data = tensor->mutable_data<float>(platform::CPUPlace());
        framework::TensorCopy(*tensor, platform::CPUPlace(), temp_tensor);
        float *w = temp_tensor->data<float>();
        paddle::distributed::Region reg(w, tensor->numel());
        regions.emplace_back(std::move(reg));
        float *origin = tensor->data<float>();
        VLOG(1) << "AsyncCommunicator::InitImpl Var " << t << " Origin_data[0] "
                << origin[0] << " Origin_data[-1] "
                << origin[tensor->numel() - 1] << " Temp_data[0] " << w[0]
                << " Temp_data[-1] " << w[tensor->numel() - 1];
#endif
      } else {
        float *w = tensor->data<float>();
        paddle::distributed::Region reg(w, tensor->numel());
        regions.emplace_back(std::move(reg));
      }
    }
  }

  if (_worker_ptr.get() == nullptr) {
    google::protobuf::TextFormat::ParseFromString(dist_desc, &_ps_param);
    init_gflag(_ps_param.init_gflags());
    server_nums = host_sign_list.size();
    _ps_env = paddle::distributed::PaddlePSEnvironment();
    _ps_env.set_ps_servers(const_cast<uint64_t *>(host_sign_list.data()),
                           server_nums);
    _worker_ptr = std::shared_ptr<paddle::distributed::PSClient>(
        paddle::distributed::PSClientFactory::create(_ps_param));
    _worker_ptr->configure(_ps_param, _dense_pull_regions, _ps_env,
                           trainer_id_);
  }
}

AsyncCommunicator::~AsyncCommunicator() {
  running_ = false;
  if (main_thread_) main_thread_->join();
  if (recv_thread_) recv_thread_->join();
}

void AsyncCommunicator::Start() {
  VLOG(1) << "Communicator start";
  if (!communicator_) {
    VLOG(0) << "Communicator is not inited, do nothing";
  } else {
    VLOG(1) << "start send thread and recv thread";
    waiting_ = true;
    running_ = true;
    // flushing_ = false;
    BarrierTriggerReset(max_merge_var_num_);
    // start send and recv thread
    main_thread_.reset(
        new std::thread(std::bind(&AsyncCommunicator::MainThread, this)));
    // recv_thread_.reset(
    //   new std::thread(std::bind(&AsyncCommunicator::RecvThread, this)));
  }
}

void AsyncCommunicator::Stop() {
  VLOG(1) << "Communicator stop";
  running_ = false;
  if (!communicator_) {
    VLOG(0) << "Communicator is not inited, do nothing";
  } else {
    if (recv_thread_) {
      VLOG(1) << "stop recv thread";
      recv_thread_->join();
      recv_thread_.reset(nullptr);
    }
    if (main_thread_) {
      VLOG(1) << "stop main thread";
      main_thread_->join();
      main_thread_.reset(nullptr);
    }
  }
  VLOG(1) << "Communicator stop done";
}

bool AsyncCommunicator::Check(const std::vector<std::string> &var_tables) {
  PADDLE_ENFORCE_EQ(
      var_tables.size(), 1,
      platform::errors::InvalidArgument("var_tables.size() == 1 is permitted"));

  auto table_name = var_tables[0];
  if (send_varname_to_ctx_.find(table_name) == send_varname_to_ctx_.end())
    return false;
  return true;
}

bool AsyncCommunicator::Check(const int table_id) {
  for (auto &iter : send_varname_to_ctx_) {
    auto &ctx = iter.second;
    if (ctx.table_id == table_id) return true;
  }
  return false;
}

void AsyncCommunicator::Send(const std::vector<std::string> &var_names,
                             const framework::Scope &scope) {
  waiting_ = false;
  for (size_t i = 0; i < var_names.size(); i++) {
    auto *var = scope.FindVar(var_names[i]);
    auto tmp_grad_var = std::make_shared<Variable>();
    framework::CopyVariable(*var, tmp_grad_var.get());
    send_varname_to_queue_[var_names[i]]->Push(tmp_grad_var);
  }
}

void AsyncCommunicator::SendSparse(const std::string &var_name, int table_id) {
  size_t request_call_num = _worker_ptr->get_server_nums();
  std::vector<uint64_t> sparse_push_keys;
  std::vector<float *> push_g_vec;

  auto *send_var = send_scope_->FindVar(var_name);
  auto *tensor = send_var->GetMutable<SelectedRows>();
  auto dim = tensor->value().dims()[1];

  std::transform(tensor->rows().begin(), tensor->rows().end(),
                 std::back_inserter(sparse_push_keys),
                 [&](int id) { return static_cast<uint64_t>(id); });

  auto *data_ptr = tensor->mutable_value()->data<float>();
  for (auto i = 0; i < static_cast<int>(sparse_push_keys.size()); ++i) {
    push_g_vec.push_back(data_ptr + i * dim);
  }

  ++_async_call_num;
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [this, request_call_num](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_response(i, PS_PUSH_SPARSE_TABLE) != 0) {
            ret = -1;
            break;
          }
        }
        closure->set_promise_value(ret);
        --_async_call_num;
      });
  _worker_ptr->push_sparse_raw_gradient(table_id, sparse_push_keys.data(),
                                        (const float **)push_g_vec.data(),
                                        sparse_push_keys.size(), closure);
  return;
}

void AsyncCommunicator::SendDense(const CommContext &ctx) {
  auto dense_data = std::make_shared<std::vector<float>>();
  size_t request_call_num = _worker_ptr->get_server_nums();
  std::vector<std::string> var_names = ctx.origin_varnames;
  uint32_t num_per_shard = dense_dim_per_shard(
      send_varname_to_ctx_[ctx.var_name].height_sections[0], request_call_num);
  dense_data->resize(num_per_shard *
                     request_call_num);  // accessor->update_dim() = 1
  float *data = dense_data->data();
  uint32_t pos = 0;

  for (size_t i = 0; i < var_names.size(); ++i) {
    const LoDTensor tensor =
        send_scope_->FindVar(var_names[i])->Get<LoDTensor>();
    size_t count = static_cast<size_t>(tensor.numel());
    const float *g = tensor.data<float>();

    CHECK(pos + count <= dense_data->size())
        << "invalid dense size, cur pos[" << pos << "]"
        << " data_num[" << count << "] size[" << dense_data->size() << "]";
    memcpy(data + pos, g, count * sizeof(float));
    pos += count;
  }

  ++_async_call_num;
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [this, request_call_num](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_response(i, PS_PUSH_DENSE_TABLE) != 0) {
            ret = -1;
            brpc::Controller *cntl = closure->cntl(i);
            VLOG(0) << "Call push dense failed: " << cntl->ErrorText();
            break;
          }
        }
        closure->set_promise_value(ret);
        --_async_call_num;
      });

  _worker_ptr->push_dense_raw_gradient(ctx.table_id, data, dense_data->size(),
                                       closure);
  return;
}

void HalfAsyncCommunicator::Clean() {
  for (auto &iter : send_varname_to_queue_) {
    auto &var_name = iter.first;
    auto &var_queue = iter.second;

    while (var_queue->Size() > 0) {
      var_queue->Pop();
    }

    VLOG(3) << "clean var: " << var_name << " done";
  }
}

void HalfAsyncCommunicator::BarrierTriggerDecrement() {
  barrier_trigger_--;
  VLOG(3) << "BarrierTriggerDecrement decrement barrier trigger to "
          << barrier_trigger_.load();
}

void HalfAsyncCommunicator::BarrierTriggerReset(int initial_val) {
  barrier_trigger_.store(initial_val);

  VLOG(3) << "BarrierTriggerReset reset barrier trigger to "
          << barrier_trigger_.load();
}

void HalfAsyncCommunicator::Barrier() {
  barrier_counter_++;

  if (!running_) {
    VLOG(3) << "Communicator is not running, release barrier";
    return;
  }

  {
    std::unique_lock<std::mutex> lk(barrier_mutex_);
    barrier_cond_.wait(lk, [this] { return (barrier_counter_ == 0); });
  }
}

int HalfAsyncCommunicator::BatchesCounter() {
  while (running_) {
    if (barrier_counter_.load() >= barrier_trigger_.load() &&
        barrier_trigger_.load() != 0) {
      break;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  return barrier_counter_.load();
}

void HalfAsyncCommunicator::SendByCommunicator() {
  int batches = BatchesCounter();
  VLOG(1) << "HalfAsyncCommunicator::BatchesCounter = " << batches;
  if (batches <= 0) return;

  for (auto &iter : send_varname_to_ctx_) {
    auto &ctx = iter.second;
    auto &varnames = ctx.origin_varnames;
    auto &table_id = ctx.table_id;
    size_t var_nums = varnames.size();

    std::vector<std::vector<std::shared_ptr<Variable>>> vars;
    vars.resize(var_nums);
    for (size_t i = 0; i < var_nums; i++) {
      auto &var_name = varnames[i];
      auto &var_queue = send_varname_to_queue_[var_name];
      for (int j = 0; j < batches; j++) vars[i].push_back(var_queue->Pop());
      MergeVars<float>(var_name, vars[i], send_scope_.get(), 1);
    }

    if (ctx.is_sparse) {
      PADDLE_ENFORCE_EQ(varnames.size(), 1, "");
      SendSparse(varnames[0], table_id);
    } else {
      // grad_num_.fetch_add(merged_var_num, std::memory_order_relaxed);
      SendDense(ctx);
    }
  }
}

void HalfAsyncCommunicator::BarrierWeakUp() {
  barrier_counter_.store(0);
  barrier_cond_.notify_all();
}

void SyncCommunicator::BarrierSend() {
  if (!running_) return;
  BarrierWithTable(0);
  VLOG(4) << "BarrierSend with SyncCommunicator";
}

void SyncCommunicator::BarrierRecv() {
  if (!running_) return;
  BarrierWithTable(1);

  VLOG(4) << "BarrierRecv with SyncCommunicator";
}

}  // namespace distributed
}  // namespace paddle
