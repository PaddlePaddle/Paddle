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

#include "paddle/fluid/distributed/ps/service/communicator/communicator.h"
#include <google/protobuf/text_format.h>
#include "gflags/gflags.h"
#include "paddle/fluid/distributed/ps/service/brpc_ps_client.h"
#include "paddle/fluid/distributed/ps/wrapper/fleet.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/string/string_helper.h"

#define LEARNING_RATE_DECAY_COUNTER "@LR_DECAY_COUNTER@"
#define STEP_COUNTER "@PS_STEP_COUNTER@"

namespace paddle {
namespace distributed {

using framework::LoDTensor;
using phi::SelectedRows;

const uint32_t MAX_FEASIGN_NUM = 1024 * 100 * 100;

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

Communicator::Communicator() {}

void Communicator::InitGFlag(const std::string &gflags) {
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
  char *flags_ptr[flags.size()];
  for (size_t i = 0; i < flags.size(); ++i) {
    flags_ptr[i] = (char *)(flags[i].c_str());  // NOLINT
  }
  int params_cnt = flags.size();
  char **params_ptr = &(flags_ptr[0]);
  ::GFLAGS_NAMESPACE::ParseCommandLineFlags(&params_cnt, &params_ptr, true);
}

std::once_flag Communicator::init_flag_;
std::shared_ptr<Communicator> Communicator::communicator_(nullptr);

void Communicator::InitBrpcClient(
    const std::string &dist_desc,
    const std::vector<std::string> &host_sign_list) {
  auto fleet = paddle::distributed::FleetWrapper::GetInstance();
  if (_worker_ptr.get() == nullptr) {
    _worker_ptr = fleet->worker_ptr_;
  }
  return;
}

std::vector<uint64_t> Communicator::GetClientInfo() {
  std::vector<uint64_t> res = _ps_env.GetClientInfo();
  for (auto rr : res) {
    VLOG(2) << "Communicator::GetClientInfo " << rr;
  }
  return res;
}

int Communicator::SetClients(std::vector<uint64_t> &host_sign_list) {
  int node = host_sign_list.size();
  return _ps_env.SetPsClients(host_sign_list.data(), node);
}

void Communicator::RpcRecvDense(const std::vector<std::string> &varnames,
                                int table_id, Scope *scope) {
  platform::RecordEvent record_event("Communicator->RpcRecvDense",
                                     platform::TracerEventType::Communication,
                                     1);
  std::vector<paddle::distributed::Region> regions;
  regions.reserve(varnames.size());
  for (auto &t : varnames) {
    Variable *var = scope->Var(t);
    LoDTensor *tensor = var->GetMutable<LoDTensor>();
    if (platform::is_gpu_place(tensor->place())) {
#ifdef PADDLE_WITH_CUDA
      Variable *temp_var = xpu_temp_scope_->Var(t);
      LoDTensor *temp_tensor = temp_var->GetMutable<LoDTensor>();
      temp_tensor->Resize(tensor->dims());
      float *temp_data = temp_tensor->mutable_data<float>(platform::CPUPlace());
      paddle::distributed::Region reg(temp_data, tensor->numel());
      regions.emplace_back(std::move(reg));
      VLOG(1) << "AsyncCommunicator::RpcRecvDense Var " << t << " table_id "
              << table_id << " Temp_data[0] " << temp_data[0]
              << " Temp_data[-1] " << temp_data[tensor->numel() - 1];
#endif
    } else {
      float *w = tensor->mutable_data<float>(tensor->place());
      paddle::distributed::Region reg(w, tensor->numel());
      regions.emplace_back(std::move(reg));
    }
  }
  auto status =
      _worker_ptr->PullDense(regions.data(), regions.size(), table_id);
  status.wait();

  for (auto &t : varnames) {
    Variable *var = scope->FindVar(t);
    LoDTensor *tensor = var->GetMutable<LoDTensor>();
    VLOG(3) << "AsyncCommunicator::RecvNoBarrier Var " << t << " On gpu? "
            << platform::is_gpu_place(tensor->place());

    float *temp_recv_data = tensor->mutable_data<float>(platform::CPUPlace());
    VLOG(3) << "AsyncCommunicator::RpcRecvDense Var " << t << " table_id "
            << table_id << " Temp_data[0] " << temp_recv_data[0]
            << " Temp_data[-1] " << temp_recv_data[tensor->numel() - 1];
    if (platform::is_gpu_place(tensor->place())) {
#ifdef PADDLE_WITH_CUDA
      LoDTensor *temp_tensor =
          xpu_temp_scope_->FindVar(t)->GetMutable<LoDTensor>();
      framework::TensorCopy(*temp_tensor, tensor->place(), tensor);
      float *temp_data = temp_tensor->mutable_data<float>(platform::CPUPlace());
      VLOG(1) << "AsyncCommunicator::RpcRecvDense Var " << t << " table_id "
              << table_id << " Temp_data[0] " << temp_data[0]
              << " Temp_data[-1] " << temp_data[tensor->numel() - 1];
#endif
    }
  }

  return;
}

void Communicator::RpcSendDenseParam(const std::vector<std::string> &varnames,
                                     int table_id, const Scope &scope) {
  platform::RecordEvent record_event("Communicator->RpcSendDenseParam",
                                     platform::TracerEventType::Communication,
                                     1);
  auto place = platform::CPUPlace();
  std::vector<paddle::distributed::Region> regions;
  for (auto &t : varnames) {
    Variable *var = scope.FindVar(t);
    CHECK(var != nullptr) << "var[" << t << "] not found";
    LoDTensor *tensor = var->GetMutable<LoDTensor>();
    if (platform::is_gpu_place(tensor->place())) {
#ifdef PADDLE_WITH_CUDA
      Variable *temp_var = xpu_temp_scope_->Var(t);
      LoDTensor *temp_tensor = temp_var->GetMutable<LoDTensor>();
      temp_tensor->Resize(tensor->dims());
      float *temp_data = temp_tensor->mutable_data<float>(platform::CPUPlace());
      framework::TensorCopy(*tensor, platform::CPUPlace(), temp_tensor);
      paddle::distributed::Region reg(temp_data, tensor->numel());
      regions.emplace_back(std::move(reg));
      VLOG(1) << "AsyncCommunicator::RpcSendDenseParam Var " << t
              << " table_id " << table_id << " Temp_data[0] " << temp_data[0]
              << " Temp_data[-1] " << temp_data[tensor->numel() - 1];
#endif
    } else {
      float *w = tensor->mutable_data<float>(place);
      paddle::distributed::Region reg(w, tensor->numel());
      regions.emplace_back(std::move(reg));
      VLOG(1) << "AsyncCommunicator::RpcSendDenseParam Var " << t
              << " talbe_id " << table_id << " Temp_data[0] " << w[0]
              << " Temp_data[-1] " << w[tensor->numel() - 1];
    }
  }
  auto status =
      _worker_ptr->PushDenseParam(regions.data(), regions.size(), table_id);
  status.wait();
  VLOG(4) << "RPC Send Dense Param " << table_id << " done!";
  return;
}

void Communicator::RpcSendDense(const CommContext &ctx, const Scope &scope) {
  platform::RecordEvent record_event("Communicator->RpcSendDense",
                                     platform::TracerEventType::Communication,
                                     1);
  auto &var_names = ctx.origin_varnames;
  auto &table_id = ctx.table_id;
  auto dense_data = std::make_shared<std::vector<float>>();
  size_t request_call_num = _worker_ptr->GetServerNums();
  uint32_t num_per_shard =
      DenseDimPerShard(ctx.height_sections[0], request_call_num);
  dense_data->resize(num_per_shard *
                     request_call_num);  // accessor->update_dim() = 1
  float *data = dense_data->data();
  uint32_t pos = 0;
  for (size_t i = 0; i < var_names.size(); ++i) {
    const LoDTensor tensor = scope.FindVar(var_names[i])->Get<LoDTensor>();
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
        auto *closure = (DownpourBrpcClosure *)done;  // NOLINT
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_response(i, PS_PUSH_DENSE_TABLE) != 0) {
            ret = -1;
            break;
          }
        }
        closure->set_promise_value(ret);
        --_async_call_num;
      });
  auto status = _worker_ptr->PushDenseRawGradient(table_id, data,
                                                  dense_data->size(), closure);
  status.wait();
  return;
}

void Communicator::RpcSendSparseParam(const std::string &varname, int table_id,
                                      const Scope &scope) {
  platform::RecordEvent record_event("Communicator->RpcSendSparseParam",
                                     platform::TracerEventType::Communication,
                                     1);
  size_t request_call_num = _worker_ptr->GetServerNums();
  std::vector<float *> push_g_vec;

  auto *send_var = scope.FindVar(varname);
  auto *tensor = send_var->GetMutable<framework::LoDTensor>();
  auto dim = tensor->dims()[1];
  uint64_t sparse_num = static_cast<uint64_t>(tensor->dims()[0]);
  std::vector<uint64_t> sparse_push_keys(sparse_num);
  std::iota(sparse_push_keys.begin(), sparse_push_keys.end(), 0);
  push_g_vec.reserve(sparse_num);

  for (auto i = 0; i < static_cast<int>(sparse_push_keys.size()); ++i) {
    push_g_vec.push_back(tensor->data<float>() + i * dim);
  }

  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [this, request_call_num](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;  // NOLINT
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_response(i, PS_PUSH_SPARSE_PARAM) != 0) {
            ret = -1;
            break;
          }
        }
        closure->set_promise_value(ret);
      });
  auto status = _worker_ptr->PushSparseParam(table_id, sparse_push_keys.data(),
                                             (const float **)push_g_vec.data(),
                                             sparse_push_keys.size(), closure);
  status.wait();
  return;
}

void Communicator::RpcSendSparse(const std::string &var_name, int table_id,
                                 const Scope &scope) {
  platform::RecordEvent record_event("Communicator->RpcSendSparse",
                                     platform::TracerEventType::Communication,
                                     1);
  size_t request_call_num = _worker_ptr->GetServerNums();
  std::vector<uint64_t> sparse_push_keys;
  std::vector<float *> push_g_vec;

  auto *send_var = scope.FindVar(var_name);
  auto *tensor = send_var->GetMutable<phi::SelectedRows>();
  auto dim = tensor->value().dims()[1];
  std::transform(tensor->rows().begin(), tensor->rows().end(),
                 std::back_inserter(sparse_push_keys),
                 [&](int64_t id) { return static_cast<uint64_t>(id); });

  for (auto i = 0; i < static_cast<int>(sparse_push_keys.size()); ++i) {
    push_g_vec.push_back(tensor->mutable_value()->data<float>() + i * dim);
  }

  // TODO(wangguanqun): padding_idx is not ignored, this is a bug.
  // if padding_idx == padding in datareader, the server will core.
  /*
  for (size_t i = 0; i < tensor->rows().size(); ++i) {
    uint64_t real_id = static_cast<uint64_t>(tensor->rows()[i]);
    if (real_id != 0) {
      sparse_push_keys.push_back(real_id);
      push_g_vec.push_back(tensor->mutable_value()->data<float>() + i * dim);
    }
  }
  */

  ++_async_call_num;
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [this, request_call_num](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;  // NOLINT
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_response(i, PS_PUSH_SPARSE_TABLE) != 0) {
            ret = -1;
            break;
          }
        }
        closure->set_promise_value(ret);
        --_async_call_num;
      });
  auto status = _worker_ptr->PushSparseRawGradient(
      table_id, sparse_push_keys.data(), (const float **)push_g_vec.data(),
      sparse_push_keys.size(), closure);
  status.wait();
  return;
}

void Communicator::RpcRecvSparse(const std::string &varname, int table_id,
                                 Scope *scope) {
  platform::RecordEvent record_event("Communicator->RpcRecvSparse",
                                     platform::TracerEventType::Communication,
                                     1);
  auto *send_var = scope->Var(varname);
  auto *tensor = send_var->GetMutable<framework::LoDTensor>();
  auto dim = tensor->dims()[1];
  uint64_t sparse_num = static_cast<uint64_t>(tensor->dims()[0]);

  std::vector<uint64_t> sparse_push_keys(sparse_num);
  std::iota(sparse_push_keys.begin(), sparse_push_keys.end(), 0);

  std::vector<float *> push_g_vec;
  for (auto i = 0; i < static_cast<int>(sparse_push_keys.size()); ++i) {
    push_g_vec.push_back(tensor->data<float>() + i * dim);
  }

  bool training = true;

  auto status = _worker_ptr->PullSparseParam(
      (float **)push_g_vec.data(), table_id,  // NOLINT
      sparse_push_keys.data(), sparse_push_keys.size(), training);
  status.wait();
  return;
}

void Communicator::InitParams(const RecvCtxMap &recv_varname_to_ctx) {
  if (trainer_id_ == 0) {
    for (auto &iter : recv_varname_to_ctx) {
      auto &table_id = iter.first;
      auto &varnames = iter.second;
      RpcSendDenseParam(varnames, table_id, *recv_scope_);
      VLOG(1) << "push dense param to table " << table_id
              << " from 0' trainer done";
    }
  }
  return;
}

void Communicator::PullDense(const RecvCtxMap &recv_varname_to_ctx) {
  for (auto &iter : recv_varname_to_ctx) {
    auto &table_id = iter.first;
    auto &varnames = iter.second;
    RpcRecvDense(varnames, table_id, recv_scope_);
    VLOG(1) << "pull dense param to table " << table_id
            << " from 0' trainer done";
  }
  return;
}

void Communicator::RpcProfilerControl() {
  if (trainer_id_ == 0) {
    if (!do_server_profiler_ && platform::IsProfileEnabled()) {
      // send profiler start flag
      do_server_profiler_ = true;
      auto start_status = _worker_ptr->StartProfiler();
      start_status.wait();
    } else if (do_server_profiler_ && !platform::IsProfileEnabled()) {
      // send profiler end flag
      auto stop_status = _worker_ptr->StopProfiler();
      stop_status.wait();
      do_server_profiler_ = false;
    }
  }
}

void Communicator::SendGlobalStep(const CommContext &ctx, int batches,
                                  Scope *send_scope) {
  if (batches == 0) {
    return;
  }
  platform::RecordEvent record_event("Communicator->SendGlobalStep",
                                     platform::TracerEventType::Communication,
                                     1);
  auto &table_id = ctx.table_id;
  size_t request_call_num = _worker_ptr->GetServerNums();

  auto &var_name = STEP_COUNTER;
  auto *out_var = send_scope->Var(var_name);
  auto *out_t = out_var->GetMutable<framework::LoDTensor>();
  auto *data = out_t->mutable_data<int64_t>({1}, platform::CPUPlace());
  data[0] = static_cast<int64_t>(batches);
  VLOG(3) << "Communicator::SendGlobalStep send: " << batches;
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [this, request_call_num](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;  // NOLINT
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_response(i, PS_PUSH_GLOBAL_STEP) != 0) {
            ret = -1;
            break;
          }
        }
        closure->set_promise_value(ret);
      });
  auto status = _worker_ptr->PushGlobalStep(table_id, data, closure);
  status.wait();
  return;
}

void AsyncCommunicator::RecvThread() {
  if (!independent_recv_) return;
  VLOG(3) << "Independent RecvThread Start and Wait";

  while (running_) {
    int grad_num = grad_num_.load();
    if (grad_num > min_send_grad_num_before_recv_) {
      RecvByCommunicator();
      grad_num_.store(0);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  VLOG(1) << "communicator stopped, independent recv thread exit";
}

void AsyncCommunicator::RecvByCommunicator() {
  if (!running_) return;
  RecvNoBarrier();
  VLOG(3) << "run recv graph end";
}

void AsyncCommunicator::RecvNoBarrier() {
  for (auto &iter : recv_varname_to_ctx_) {
    auto &table_id = iter.first;
    auto &varnames = iter.second;
    RpcRecvDense(varnames, table_id, recv_scope_);
  }

  for (auto &iter : recv_varname_to_ctx_) {
    auto var_names = iter.second;
    for (auto &t : var_names) {
      Variable *var = recv_scope_->FindVar(t);
      LoDTensor *tensor = var->GetMutable<LoDTensor>();
      VLOG(3) << "AsyncCommunicator::RecvNoBarrier Var " << t << " On gpu? "
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
  std::vector<std::future<void>> tasks;
  tasks.reserve(send_varname_to_ctx_.size());

  for (auto &iter : send_varname_to_ctx_) {
    auto &ctx = iter.second;

    auto send_recv_task = [this, &ctx] {
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
      if (merged_var_num == 0) return;

      for (size_t i = 0; i < var_nums; i++) {
        auto &var_name = varnames[i];
        if (var_name == STEP_COUNTER) {
          MergeVars<int64_t>(var_name, vars[i], send_scope_.get(), 1);
        } else {
          MergeVars<float>(var_name, vars[i], send_scope_.get(), 1);
        }
      }

      if (ctx.is_tensor_table) {
        SendGlobalStep(ctx, merged_var_num, send_scope_.get());
      } else if (ctx.is_sparse) {
        PADDLE_ENFORCE_EQ(
            varnames.size(), 1,
            platform::errors::InvalidArgument(
                "sparse variables can only be merged by one variables"));
        RpcSendSparse(varnames[0], table_id, *send_scope_);
      } else {
        RpcSendDense(ctx, *send_scope_);
        if (!independent_recv_ &&
            recv_varname_to_ctx_.find(table_id) != recv_varname_to_ctx_.end()) {
          auto recv_varnames = recv_varname_to_ctx_.at(table_id);
          RpcRecvDense(recv_varnames, table_id, recv_scope_);
        }
      }
      if (independent_recv_) {
        grad_num_.fetch_add(1, std::memory_order_relaxed);
      }
    };
    tasks.emplace_back(send_threadpool_->enqueue(std::move(send_recv_task)));
  }
  for (auto &task : tasks) {
    task.wait();
  }
  return;
}

void AsyncCommunicator::PushDensePostProcessing() {
  if (independent_recv_) {
    grad_num_.fetch_add(1, std::memory_order_relaxed);
  }
  return;
}

void AsyncCommunicator::MainThread() {
  VLOG(3) << "AsyncCommunicator MainThread start and wait";

  while (waiting_ && running_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    VLOG(3) << "wait for running";
  }

  while (running_) {
    SendByCommunicator();
    RpcProfilerControl();
  }
  VLOG(1) << "communicator stopped, send thread exit";
}

void AsyncCommunicator::PullSparseToTensorSync(
    const uint64_t table_id, int fea_dim, uint64_t padding_id,
    platform::Place place, bool is_training,
    std::vector<const LoDTensor *> *inputs, std::vector<LoDTensor *> *outputs) {
  std::vector<uint64_t> fea_keys;
  std::vector<float *> pull_result_ptr;
  fea_keys.reserve(MAX_FEASIGN_NUM / 100);
  pull_result_ptr.reserve(MAX_FEASIGN_NUM / 100);
  std::vector<float> init_value(fea_dim, 0);
  framework::LoDTensor *output = nullptr;
  float *output_data = nullptr;
  size_t output_index = -1;
  size_t output_len = 0;
  for (size_t index = 0; index < inputs->size(); ++index) {
    const framework::LoDTensor *tensor = inputs->at(index);
    const int64_t *ids = tensor->data<int64_t>();
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
      _worker_ptr->PullSparse(pull_result_ptr.data(), table_id, fea_keys.data(),
                              fea_keys.size(), is_training);
  status.wait();
  auto ret = status.get();
  if (ret != 0) {
    LOG(ERROR) << "fleet pull sparse failed, status[" << ret << "]";
    sleep(sleep_seconds_before_fail_exit_);
  }
}

void AsyncCommunicator::PushSparseFromTensorAsync(
    const uint64_t table_id, int fea_dim, uint64_t padding_id,
    platform::Place place, std::vector<const framework::LoDTensor *> *inputs,
    const framework::LoDTensor *shows, const framework::LoDTensor *clks,
    std::vector<framework::LoDTensor *> *outputs) {
  int batch_size = -1;
  bool batch_size_consist = true;
  for (auto *input : *inputs) {
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

  VLOG(2) << "fleet.cc::emb_dim: " << fea_dim << " batch_size: " << batch_size
          << " batch_size_consist: " << batch_size_consist;

  // TODO(zhaocaibei123): check type of show/clk is int? float? uint64?
  // const long int* show_tensor = shows->data<int64_t>();
  // const long int* clk_tensor = clks->data<int64_t>();
  const int64_t *show_tensor = shows->data<int64_t>();
  const int64_t *clk_tensor = clks->data<int64_t>();

  for (size_t index = 0; index < inputs->size(); ++index) {
    framework::LoDTensor *g_tensor = outputs->at(index);
    float *g = g_tensor->data<float>();

    if (batch_size_consist) {  // TODO(zhaocaibei123): add config
                               // scale_sparse_gradient_with_batch_size_
      Eigen::Map<
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          g_mat(g, g_tensor->numel() / fea_dim, fea_dim);
      g_mat.rightCols(fea_dim - 2) *=
          batch_size;  // hard code here, because of cvm_grad op
    }

    const framework::LoDTensor *tensor = inputs->at(index);
    const int64_t *ids = tensor->data<int64_t>();
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
          push_values.emplace_back(fea_dim + 1);
          // slot show clk grad... consistent with CtrCommonPushValue defined in
          // ctr_accessor.h
          push_values.back()[0] = 2;  // TODO(zhaocaibei123): slot
          // push_values.back()[1] =
          //    (i >= show_size ? 1 : static_cast<float>(show_tensor[i]));
          // push_values.back()[2] =
          //    (i >= clk_size ? 0 : static_cast<float>(clk_tensor[i]));

          float *data = push_values.back().data() + 1;  // hard code here

          memcpy(data, g + output_len, sizeof(float) * fea_dim);

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
        push_values.emplace_back(fea_dim + 1);
        // slot show clk grad... consistent with CtrCommonPushValue defined in
        // ctr_accessor.h
        push_values.back()[0] = 2;  // TODO(zhaocaibei123): slot
        // push_values.back()[1] =
        //    (i >= show_size ? 1 : static_cast<float>(show_tensor[i]));
        // push_values.back()[2] =
        //    (i >= clk_size ? 0 : static_cast<float>(clk_tensor[i]));

        float *data = push_values.back().data() + 1;

        memcpy(data, g + output_len, sizeof(float) * fea_dim);

        ++input_idx;
      }
    }
    CHECK(output_len == g_tensor->numel());
  }

  std::vector<float *> push_g_vec(input_idx, nullptr);

  for (auto i = 0u; i < push_keys.size(); ++i) {
    push_g_vec[i] = push_values.at(i).data();
  }

  PADDLE_ENFORCE_EQ(
      this->Check(table_id), true,
      platform::errors::InvalidArgument(
          "can not find table: %s, please check your config", table_id));
  auto status = _worker_ptr->PushSparse(table_id, push_keys.data(),
                                        (const float **)push_g_vec.data(),
                                        push_keys.size());
}

void HalfAsyncCommunicator::MainThread() {
  VLOG(3) << "HalfAsyncCommunicator MainThread start and wait";

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
                                 Scope *recv_scope) {
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
  send_threadpool_.reset(new ::ThreadPool(thread_pool_size_));
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
    if (independent_recv_) {
      recv_thread_.reset(
          new std::thread(std::bind(&AsyncCommunicator::RecvThread, this)));
    }
  }
}

void AsyncCommunicator::Stop() {
  VLOG(1) << "Communicator stop begin";
  running_ = false;
  if (!communicator_) {
    VLOG(0) << "Communicator is not inited, do nothing";
  } else {
    // _worker_ptr->FinalizeWorker();
    VLOG(1) << "client finalize_worker done";
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
  if (send_varname_to_ctx_.find(table_name) == send_varname_to_ctx_.end()) {
    return false;
  }
  if (table_name == STEP_COUNTER) {
    VLOG(3) << "send step_counter into queue";
    auto tmp_var = std::make_shared<Variable>();
    auto *tensor = tmp_var->GetMutable<framework::LoDTensor>();
    tensor->Resize(phi::make_ddim({1}));
    auto *out_d = tensor->mutable_data<int64_t>(platform::CPUPlace());
    out_d[0] = 1;
    send_varname_to_queue_[table_name]->Push(tmp_var);
  }
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

  std::vector<std::future<void>> tasks;
  tasks.reserve(send_varname_to_ctx_.size());

  for (auto &iter : send_varname_to_ctx_) {
    auto &ctx = iter.second;
    auto send_recv_task = [this, &ctx, batches] {
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
        PADDLE_ENFORCE_EQ(
            varnames.size(), 1,
            platform::errors::InvalidArgument(
                "sparse variables can only be merged by one variables"));
        RpcSendSparse(varnames[0], table_id, *send_scope_);
      } else {
        RpcSendDense(ctx, *send_scope_);
      }
    };
    tasks.emplace_back(send_threadpool_->enqueue(std::move(send_recv_task)));
  }
  for (auto &task : tasks) {
    task.wait();
  }
  return;
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

void GeoCommunicator::Send(const std::vector<std::string> &var_names,
                           const framework::Scope &scope) {
  platform::RecordEvent record_event(
      "GeoCommunicator->Send", platform::TracerEventType::Communication, 1);
  waiting_ = false;
  auto before_send = GetCurrentUS();
  auto table_name = var_names[0];

  size_t splited_var_nums =
      send_varname_to_ctx_[table_name].splited_varnames.size();

  std::unordered_map<std::string, std::unordered_set<int64_t>> ids_table;

  for (size_t j = 0; j < splited_var_nums; j++) {
    ids_table.insert(std::pair<std::string, std::unordered_set<int64_t>>(
        send_varname_to_ctx_[table_name].splited_varnames[j],
        std::unordered_set<int64_t>()));
  }

  auto *var = scope.FindVar(table_name);

  PADDLE_ENFORCE_EQ(var->IsType<phi::SelectedRows>(), true,
                    platform::errors::InvalidArgument(
                        "Only need to send Sparse Grad in Geo mode."));
  auto &rows = var->Get<phi::SelectedRows>().rows();

  // insert ids which has not been record
  for (size_t j = 0; j < rows.size(); j++) {
    auto ep_idx = rows[j] % splited_var_nums;
    ids_table.at(send_varname_to_ctx_[table_name].splited_varnames[ep_idx])
        .insert(rows[j]);
  }

  for (auto &iter : ids_table) {
    auto &key = iter.first;
    auto &sparse_ids_set = iter.second;
    auto sparse_ids_vec = std::make_shared<std::vector<int64_t>>();
    sparse_ids_vec->assign(sparse_ids_set.begin(), sparse_ids_set.end());
    sparse_id_queues_.at(key)->Put(sparse_ids_vec);
    VLOG(3) << "push " << sparse_ids_vec->size() << " ids to " << key
            << "'s queue";
  }

  auto after_send = GetCurrentUS();
  VLOG(2) << "run send op finish. use time " << (after_send - before_send);
}

void GeoCommunicator::InitImpl(const RpcCtxMap &send_varname_to_ctx,
                               const RecvCtxMap &recv_varname_to_ctx,
                               Scope *recv_scope) {
  send_varname_to_ctx_ = std::move(send_varname_to_ctx);
  recv_varname_to_ctx_ = std::move(recv_varname_to_ctx);
  recv_scope_ = std::move(recv_scope);

  PADDLE_ENFORCE_GT(
      send_varname_to_ctx.size(), 0,
      platform::errors::InvalidArgument("send var contexts can not be zero"));

  for (auto &iter : send_varname_to_ctx_) {
    auto &ctx = iter.second;
    if (!ctx.is_sparse) {
      parallel_task_nums_ += 1;
      continue;
    }
    auto &varnames = ctx.origin_varnames;
    PADDLE_ENFORCE_EQ(
        varnames.size(), 1,
        platform::errors::InvalidArgument(
            "sparse variables can only be merged by one variables"));
    for (auto &splited_var : ctx.splited_varnames) {
      parallel_task_nums_ += 1;
      sparse_id_queues_.insert(
          std::pair<std::string, paddle::framework::Channel<
                                     std::shared_ptr<std::vector<int64_t>>>>(
              splited_var,
              paddle::framework::MakeChannel<
                  std::shared_ptr<std::vector<int64_t>>>(send_queue_size_)));
    }
  }

  send_threadpool_.reset(new ::ThreadPool(thread_pool_size_));

  delta_scope_.reset(new Scope());
  old_scope_.reset(new Scope());
  pserver_scope_.reset(new Scope());
}

void GeoCommunicator::InitParams(const RecvCtxMap &recv_varname_to_ctx) {
  std::vector<std::future<void>> tasks;
  tasks.reserve(recv_varname_to_ctx_.size());

  for (auto &iter : recv_varname_to_ctx_) {
    auto &table_id = iter.first;
    auto &varnames = iter.second;

    auto recv_task = [this, &table_id, &varnames] {
      InitDense(varnames, table_id);
    };
    tasks.emplace_back(send_threadpool_->enqueue(std::move(recv_task)));
  }

  for (auto &task : tasks) {
    task.wait();
  }

  for (auto &iter : send_varname_to_ctx_) {
    auto &ctx = iter.second;
    if (!ctx.is_sparse) continue;
    auto &varname = ctx.origin_varnames[0];
    auto &table_id = ctx.table_id;
    auto param = varname.substr(0, varname.size() - 5);
    InitSparse(param, table_id);
  }
  return;
}

void GeoCommunicator::InitDense(std::vector<std::string> &varnames,
                                int table_id) {
  if (trainer_id_ == 0) {
    RpcSendDenseParam(varnames, table_id, *recv_scope_);
    BarrierWithTable(1);
    VLOG(1) << "push dense param to table " << table_id
            << " from 0' trainer done";
  } else {
    BarrierWithTable(1);
    RpcRecvDense(varnames, table_id, recv_scope_);
    VLOG(1) << "pull dense param to table " << table_id
            << " from 0' trainer done";
  }

  // copy to old_scope
  for (auto &t : varnames) {
    auto *global_var = recv_scope_->FindVar(t);
    global_var->GetMutable<framework::LoDTensor>();
    auto *old_var = old_scope_->Var(t);
    old_var->GetMutable<framework::LoDTensor>();
    framework::CopyVariable(*global_var, old_var);
    // init pserver_scope_
    auto *pserver_var = pserver_scope_->Var(t);
    pserver_var->GetMutable<framework::LoDTensor>();
    framework::CopyVariable(*global_var, pserver_var);
  }
  VLOG(1) << "init dense table " << table_id << " done";
}

void GeoCommunicator::SendDense(const CommContext &send_ctx) {
  platform::RecordEvent record_event("GeoCommunicator->SendDense",
                                     platform::TracerEventType::Communication,
                                     1);
  auto &var_names = send_ctx.origin_varnames;
  auto &table_id = send_ctx.table_id;
  for (auto &varname : var_names) {
    auto param_name = GradToParam(varname);
    auto *var_latest = recv_scope_->FindVar(param_name);
    auto *var_timestamp = old_scope_->FindVar(param_name);

    PADDLE_ENFORCE_EQ(var_latest->IsInitialized(), true,
                      platform::errors::Unavailable(
                          "%s is not initialized, please check", param_name));
    PADDLE_ENFORCE_EQ(var_timestamp->IsInitialized(), true,
                      platform::errors::Unavailable(
                          "%s is not initialized, please check", param_name));

    auto &t_latest = var_latest->Get<framework::LoDTensor>();
    auto t_timestamp = var_timestamp->GetMutable<framework::LoDTensor>();

    paddle::platform::CPUDeviceContext cpu_ctx;
    auto *var_delta = delta_scope_->Var(varname);
    auto *t_delta = var_delta->GetMutable<framework::LoDTensor>();
    t_delta->mutable_data<float>(t_latest.dims(), cpu_ctx.GetPlace());

    auto blas = phi::funcs::GetBlas<platform::CPUDeviceContext, float>(cpu_ctx);
    blas.VSUB(t_latest.numel(), t_latest.data<float>(),
              t_timestamp->data<float>(), t_delta->data<float>());

    float coefficient = 1.0 / static_cast<float>(trainers_);
    blas.SCAL(t_latest.numel(), coefficient, t_delta->data<float>());

    blas.VADD(t_latest.numel(), t_timestamp->data<float>(),
              t_delta->data<float>(), t_timestamp->data<float>());
  }
  RpcSendDense(send_ctx, *delta_scope_);
  VLOG(1) << "Finish Send Dense " << var_names[0] << ", table_id: " << table_id;
  return;
}

void GeoCommunicator::RecvDense(const CommContext &send_ctx) {
  platform::RecordEvent record_event("GeoCommunicator->RecvDense",
                                     platform::TracerEventType::Communication,
                                     1);
  auto &table_id = send_ctx.table_id;
  auto &varnames = recv_varname_to_ctx_.at(table_id);
  // 1. recv from pserver
  RpcRecvDense(varnames, table_id, pserver_scope_.get());

  // 2.1 pserver - old => delta; 2.2 latest + old => latest 2.3 old => pserver
  paddle::platform::CPUDeviceContext cpu_ctx;
  for (auto &varname : varnames) {
    auto *var_latest = recv_scope_->FindVar(varname);
    auto t_latest = var_latest->GetMutable<framework::LoDTensor>();

    auto *var_old = old_scope_->FindVar(varname);
    auto t_old = var_old->GetMutable<framework::LoDTensor>();

    auto *var_pserver = pserver_scope_->FindVar(varname);
    auto t_pserver = var_pserver->Get<framework::LoDTensor>();

    auto *var_delta = delta_scope_->Var(varname);
    auto *t_delta = var_delta->GetMutable<framework::LoDTensor>();
    t_delta->mutable_data<float>(t_latest->dims(), cpu_ctx.GetPlace());

    auto blas = phi::funcs::GetBlas<platform::CPUDeviceContext, float>(cpu_ctx);
    blas.VSUB(t_latest->numel(), t_pserver.data<float>(), t_old->data<float>(),
              t_delta->data<float>());
    blas.VADD(t_latest->numel(), t_latest->data<float>(),
              t_delta->data<float>(), t_latest->data<float>());
    blas.VCOPY(t_latest->numel(), t_pserver.data<float>(),
               t_old->data<float>());
  }
  VLOG(1) << "Finish Recv Dense " << varnames[0] << ", table_id: " << table_id;
  return;
}

void GeoCommunicator::InitSparse(const std::string &var_name, int table_id) {
  VLOG(1) << "Init Sparse " << var_name << " : table " << table_id << " begin.";
  if (trainer_id_ == 0) {
    RpcSendSparseParam(var_name, table_id, *recv_scope_);
    BarrierWithTable(1);
    VLOG(1) << "push sparse param to table " << table_id
            << " from 0' trainer done";
  } else {
    BarrierWithTable(1);
    RpcRecvSparse(var_name, table_id, recv_scope_);
    VLOG(1) << "pull sparse param to table " << table_id
            << " from 0' trainer done";
  }

  VLOG(1) << "Init Sparse " << var_name << " : table " << table_id << " done.";
  auto *global_var = recv_scope_->FindVar(var_name);
  auto *var = old_scope_->Var(var_name);
  framework::CopyVariable(*global_var, var);
  return;
}

std::vector<int64_t> GeoCommunicator::MergeSparseIds(
    const std::string &send_varname) {
  platform::RecordEvent record_event("GeoCommunicator->MergeSparseIds",
                                     platform::TracerEventType::Communication,
                                     1);
  size_t merge_num = 0, wait_times = 0;
  std::unordered_set<int64_t> sparse_ids;
  while (merge_num < static_cast<size_t>(max_merge_var_num_)) {
    VLOG(3) << "Merge Number of " << send_varname << " = " << merge_num;
    if (sparse_id_queues_.at(send_varname)->Size() > 0) {
      wait_times = 0;
      std::shared_ptr<std::vector<int64_t>> pop_ids = nullptr;
      sparse_id_queues_.at(send_varname)->Get(pop_ids);
      for (size_t j = 0; j < pop_ids->size(); j++) {
        sparse_ids.insert(pop_ids->at(j));
      }
      merge_num += 1;
      VLOG(3) << "sparse_id_queues_(" << send_varname << ") pushed";
    } else if (sparse_id_queues_.at(send_varname)->Size() == 0) {
      VLOG(3) << "wait_times -> " << wait_times;
      if (wait_times >= static_cast<size_t>(send_wait_times_)) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      wait_times++;
      continue;
    }
  }
  std::vector<int64_t> res;
  res.assign(sparse_ids.begin(), sparse_ids.end());
  return res;
}

void GeoCommunicator::SendSparse(const std::string &varname,
                                 std::vector<int64_t> &sparse_ids, int table_id,
                                 int ep_idx) {
  platform::RecordEvent record_event("GeoCommunicator->SendSparse",
                                     platform::TracerEventType::Communication,
                                     1);
  if (sparse_ids.size() == 0) {
    return;
  }
  std::string param_name = SplitedGradToParam(varname);
  VLOG(1) << "In GeoCommunicator::SendSparse(" << varname << " " << param_name
          << ", ids.size = " << sparse_ids.size() << ", table_id: " << table_id
          << ", ep_idx: " << ep_idx;

  auto *var_latest = recv_scope_->FindVar(param_name);
  auto *var_old = old_scope_->FindVar(param_name);

  PADDLE_ENFORCE_EQ(var_latest->IsInitialized(), true,
                    platform::errors::Unavailable(
                        "%s is not initialized, please check", param_name));
  PADDLE_ENFORCE_EQ(var_old->IsInitialized(), true,
                    platform::errors::Unavailable(
                        "%s is not initialized, please check", param_name));

  auto &t_latest = var_latest->Get<framework::LoDTensor>();
  auto *t_old = var_old->GetMutable<framework::LoDTensor>();

  auto dims1 = t_latest.dims()[1];
  paddle::platform::CPUDeviceContext cpu_ctx;

  auto *var_delta = delta_scope_->Var(varname);
  auto *t_delta = var_delta->GetMutable<phi::SelectedRows>();
  auto *var_t_value = t_delta->mutable_value();
  var_t_value->Resize({static_cast<int64_t>(sparse_ids.size()), dims1});
  auto *t_value = var_t_value->mutable_data<float>(cpu_ctx.GetPlace());

  t_delta->set_rows(sparse_ids);
  t_delta->set_height(t_latest.dims()[0]);

  auto blas = phi::funcs::GetBlas<platform::CPUDeviceContext, float>(cpu_ctx);
  float coefficient = 1.0 / static_cast<float>(trainers_);

  std::vector<float *> push_g_vec;
  for (auto j = 0; j < static_cast<int>(sparse_ids.size()); ++j) {
    blas.VSUB(dims1, t_latest.data<float>() + sparse_ids[j] * dims1,
              t_old->data<float>() + sparse_ids[j] * dims1,
              t_value + j * dims1);
    blas.SCAL(dims1, coefficient, t_value + j * dims1);
    blas.VADD(dims1, t_old->data<float>() + sparse_ids[j] * dims1,
              t_value + j * dims1,
              t_old->data<float>() + sparse_ids[j] * dims1);
    push_g_vec.push_back(t_value + j * dims1);

    VLOG(5) << "DEBUG GeoCommunicator::SendSparse send sparse key "
            << sparse_ids[j] << " value[0] " << push_g_vec[j][0]
            << " value[-1] " << push_g_vec[j][dims1 - 1];
  }

  ++_async_call_num;
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(1, [this](void *done) {
    int ret = 0;
    auto *closure = (DownpourBrpcClosure *)done;  // NOLINT
    if (closure->check_response(0, PS_PUSH_SPARSE_TABLE) != 0) {
      ret = -1;
    }
    closure->set_promise_value(ret);
    --_async_call_num;
  });
  auto status = _worker_ptr->PushSparseRawGradientPartial(
      table_id, (const uint64_t *)sparse_ids.data(),
      (const float **)push_g_vec.data(), sparse_ids.size(), closure, ep_idx);
  status.wait();

  VLOG(1) << "Finish Send Sparse " << varname
          << ", ids.size = " << sparse_ids.size() << ", table_id: " << table_id;
  return;
}

void GeoCommunicator::RecvSparse(const std::string &varname, int table_id,
                                 int ep_idx) {
  platform::RecordEvent record_event("GeoCommunicator->RecvSparse",
                                     platform::TracerEventType::Communication,
                                     1);
  // 1. recv from pserver
  std::vector<uint64_t> keys;
  std::vector<float> values;
  auto status = _worker_ptr->PullGeoParam(table_id, &values, &keys, ep_idx);
  status.wait();

  std::string param = SplitedGradToParam(varname);
  VLOG(1) << "RecvSparse receive var: " << varname << " " << param << ", "
          << table_id << "; ids Size: " << keys.size()
          << "; values size: " << values.size();

  auto *var_latest = recv_scope_->FindVar(param);
  auto *var_old = old_scope_->FindVar(param);

  auto *t_latest = var_latest->GetMutable<framework::LoDTensor>();
  auto *t_old = var_old->GetMutable<framework::LoDTensor>();

  auto dims1 = t_latest->dims()[1];
  auto numel = keys.size() * dims1;

  std::vector<float> v_delta;
  v_delta.resize(numel);

  paddle::platform::CPUDeviceContext cpu_ctx;
  auto blas = phi::funcs::GetBlas<platform::CPUDeviceContext, float>(cpu_ctx);

  for (auto j = 0; j < static_cast<int>(keys.size()); ++j) {
    VLOG(5) << "DEBUG GeoCommunicator::RecvSparse recv sparse key" << keys[j]
            << "value[0] " << values[j * dims1] << " value[-1] "
            << values[j * dims1 + dims1 - 1];
    float *latest_data = t_latest->data<float>() + keys[j] * dims1;
    float *old_data = t_old->data<float>() + keys[j] * dims1;
    // pserver - old => delta
    blas.VSUB(dims1, values.data() + j * dims1, old_data,
              v_delta.data() + j * dims1);
    // latest + delta => latest
    blas.VADD(dims1, latest_data, v_delta.data() + j * dims1, latest_data);
    // pserver => old
    blas.VCOPY(dims1, values.data() + j * dims1, old_data);
  }
  VLOG(1) << "Finish Recv Sparse " << param << ", table_id: " << table_id;
}

void GeoCommunicator::MainThread() {
  VLOG(3) << "MainThread start and wait";

  while (waiting_ && running_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    VLOG(3) << "wait for running";
  }

  while (running_) {
    std::vector<std::future<void>> tasks;
    tasks.reserve(parallel_task_nums_);

    for (auto &iter : send_varname_to_ctx_) {
      auto &ctx = iter.second;
      auto &varnames = ctx.origin_varnames;
      auto &table_id = ctx.table_id;

      if (ctx.is_sparse) {
        PADDLE_ENFORCE_EQ(
            varnames.size(), 1,
            platform::errors::InvalidArgument(
                "sparse variables can only be merged by one variables"));
        int pserver_num = static_cast<int>(ctx.epmap.size());
        for (int ep_idx = 0; ep_idx < pserver_num; ep_idx++) {
          // varname: emb@GRAD, param_name: emb, splited_varname: emb.delta0
          auto send_recv_task = [this, table_id, ep_idx, &ctx] {
            auto splited_varname = ctx.splited_varnames[ep_idx];
            auto sparse_ids = MergeSparseIds(splited_varname);
            SendSparse(splited_varname, sparse_ids, table_id, ep_idx);
            RecvSparse(splited_varname, table_id, ep_idx);
          };
          tasks.emplace_back(
              send_threadpool_->enqueue(std::move(send_recv_task)));
        }
      } else {
        auto send_recv_task = [this, &ctx] {
          SendDense(ctx);
          RecvDense(ctx);
        };
        tasks.emplace_back(
            send_threadpool_->enqueue(std::move(send_recv_task)));
      }
    }
    for (auto &task : tasks) {
      task.wait();
    }
  }
}

}  // namespace distributed
}  // namespace paddle
