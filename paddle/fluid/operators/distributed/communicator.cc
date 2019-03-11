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

#include "paddle/fluid/operators/distributed/communicator.h"

#include <gflags/gflags.h>
#include <chrono>  // NOLINT
#include <thread>  // NOLINT

#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/distributed/parameter_recv.h"
#include "paddle/fluid/operators/distributed/parameter_send.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

DEFINE_bool(communicator_independent_recv_thread, true,
            "use an independent to recv vars from parameter server");
DEFINE_int32(communicator_send_queue_size, 20,
             "queue size to recv gradient before send");
DEFINE_int32(communicator_recv_wait_ms, 200, "wait time between each recv");
DEFINE_int32(communicator_thread_pool_size, 5, "thread num to do send or recv");
DEFINE_int32(communicator_max_merge_var_num, 20,
             "max var num to merge and send");
DEFINE_bool(communicator_fake_rpc, false,
            "fake mode does not really send any thing");

namespace paddle {
namespace operators {
namespace distributed {

static inline void MergeVars(const std::string &var_name,
                             const std::vector<std::shared_ptr<Variable>> &vars,
                             Scope *scope) {
  VLOG(3) << "merge " << vars.size() << " vars " << var_name << " to 1";
  PADDLE_ENFORCE(!vars.empty(), "should have value to merge!");
  auto cpu_place = platform::CPUPlace();
  auto &var0 = vars[0];
  auto *out_var = scope->Var(var_name);
  if (var0->IsType<framework::LoDTensor>()) {
    auto *out_t = out_var->GetMutable<framework::LoDTensor>();
    auto *out_ptr = out_t->mutable_data<float>(
        var0->Get<framework::LoDTensor>().dims(), cpu_place);
    auto numel = out_t->numel();
    for (auto i = 0; i < numel; ++i) {
      out_ptr[i] = 0;
      for (auto &var : vars) {
        auto &var_t = var->Get<framework::LoDTensor>();
        PADDLE_ENFORCE_EQ(var_t.numel(), numel, "should have the same dims");
        out_ptr[i] += var_t.data<float>()[i];
      }
    }
  } else if (var0->IsType<framework::SelectedRows>()) {
    auto *out_slr = out_var->GetMutable<framework::SelectedRows>();
    out_slr->mutable_rows()->clear();
    out_slr->mutable_value()->mutable_data<float>({{}}, cpu_place);
    std::vector<const paddle::framework::SelectedRows *> inputs;
    inputs.reserve(vars.size());
    for (auto &var : vars) {
      inputs.push_back(&var->Get<framework::SelectedRows>());
    }
    math::scatter::MergeAdd<paddle::platform::CPUDeviceContext, float>
        merge_add;
    auto dev_ctx = paddle::platform::CPUDeviceContext();
    merge_add(dev_ctx, inputs, out_slr, false);
  } else {
    PADDLE_THROW("unsupported var type!");
  }
}

std::unique_ptr<Communicator> Communicator::communicator_(nullptr);
std::once_flag Communicator::init_flag_;

Communicator::Communicator(const RpcCtxMap &send_varname_to_ctx,
                           const RpcCtxMap &recv_varname_to_ctx,
                           Scope *recv_scope)
    : send_varname_to_ctx_(send_varname_to_ctx),
      recv_varname_to_ctx_(recv_varname_to_ctx),
      recv_scope_(recv_scope) {
  // get all send information from graph, build vars_to_send
  VLOG(0) << "communicator_independent_recv_thread: "
          << FLAGS_communicator_independent_recv_thread;
  VLOG(0) << "communicator_send_queue_size: "
          << FLAGS_communicator_send_queue_size;
  VLOG(0) << "communicator_recv_wait_ms: " << FLAGS_communicator_recv_wait_ms;
  VLOG(0) << "communicator_thread_pool_size: "
          << FLAGS_communicator_thread_pool_size;
  VLOG(0) << "communicator_max_merge_var_num: "
          << FLAGS_communicator_max_merge_var_num;
  VLOG(0) << "communicator_fake_rpc: " << FLAGS_communicator_fake_rpc;
  send_scope_.reset(new Scope());
  for (auto &iter : send_varname_to_ctx_) {
    send_varname_to_queue_[iter.first] =
        std::make_shared<BlockingQueue<std::shared_ptr<Variable>>>(
            FLAGS_communicator_send_queue_size);
  }
  send_threadpool_.reset(new ::ThreadPool(FLAGS_communicator_thread_pool_size));
  recv_threadpool_.reset(new ::ThreadPool(FLAGS_communicator_thread_pool_size));
}

Communicator::~Communicator() {
  VLOG(3) << "~Communicator";
  running_ = false;
  if (send_thread_) send_thread_->join();
  if (recv_thread_) recv_thread_->join();
  VLOG(3) << "~Communicator done";
}

void Communicator::SendThread() {
  VLOG(3) << "SendThread start!";
  while (running_) {
    std::vector<std::future<void>> task_futures;
    task_futures.reserve(send_varname_to_ctx_.size());
    VLOG(3) << "run send graph";
    for (auto &iter : send_varname_to_queue_) {
      auto &var_name = iter.first;
      auto &var_queue = iter.second;
      if (var_queue->Size() > 0) {
        auto send_task = [this, &var_name, &var_queue] {
          VLOG(3) << "merge var " << var_name << " and send";
          std::vector<std::shared_ptr<Variable>> vars;
          size_t merged_var_num = 0;
          while (var_queue->Size() > 0 &&
                 merged_var_num < FLAGS_communicator_max_merge_var_num) {
            vars.push_back(var_queue->Pop());
            merged_var_num++;
          }
          MergeVars(var_name, vars, send_scope_.get());
          auto send_functor = distributed::ParameterSend<float>();
          auto &ctx = send_varname_to_ctx_.at(var_name);
          if (!FLAGS_communicator_fake_rpc) {
            send_functor(ctx, *send_scope_, true);
          }
        };
        task_futures.emplace_back(
            send_threadpool_->enqueue(std::move(send_task)));
      } else {
        VLOG(3) << var_name << " queue empty";
      }
    }
    for (auto &task_f : task_futures) {
      task_f.wait();
    }
    VLOG(3) << "run send graph done";
    if (!FLAGS_communicator_independent_recv_thread) {
      RecvAll();
    }
  }
}

void Communicator::RecvAll() {
  VLOG(3) << "parallel run recv graph";
  std::vector<std::future<void>> task_futures;
  task_futures.reserve(recv_varname_to_ctx_.size());
  for (auto &iter : recv_varname_to_ctx_) {
    auto recv_task = [this, &iter] {
      auto &var_name = iter.first;
      VLOG(3) << "recv var " << var_name;
      auto recv_functor = distributed::ParameterRecv<float>();
      if (!FLAGS_communicator_fake_rpc) {
        recv_functor(iter.second, *recv_scope_);
      }
    };
    task_futures.emplace_back(recv_threadpool_->enqueue(std::move(recv_task)));
  }
  for (auto &task : task_futures) {
    task.wait();
  }
  VLOG(3) << "run recv graph done";
}

void Communicator::RecvThread() {
  VLOG(3) << "RecvThread start!";
  while (running_) {
    RecvAll();
    std::this_thread::sleep_for(
        std::chrono::milliseconds(FLAGS_communicator_recv_wait_ms));
  }
}

void Communicator::Send(const std::string &var_name,
                        const framework::Scope &scope) {
  VLOG(3) << "communicator send " << var_name;
  // push var into send queue by var_name
  auto *grad_var = scope.FindVar(var_name);
  PADDLE_ENFORCE(grad_var->IsInitialized(), "grad var should be inited");
  auto tmp_grad_var = std::make_shared<Variable>();
  framework::CopyVariable(*grad_var, tmp_grad_var.get());
  auto &queue = send_varname_to_queue_.at(var_name);
  VLOG(3) << "send " << var_name << " queue size " << queue->Size();
  queue->Push(tmp_grad_var);
}

Communicator *Communicator::GetInstance() { return communicator_.get(); }

void Communicator::Start() {
  running_ = true;
  // start send and recv thread
  send_thread_.reset(
      new std::thread(std::bind(&Communicator::SendThread, this)));
  if (FLAGS_communicator_independent_recv_thread) {
    recv_thread_.reset(
        new std::thread(std::bind(&Communicator::RecvThread, this)));
  }
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
