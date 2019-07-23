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
#include <paddle/fluid/framework/program_desc.h>
#include <chrono>  // NOLINT
#include <thread>  // NOLINT

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/distributed/parameter_recv.h"
#include "paddle/fluid/operators/distributed/parameter_send.h"

DEFINE_bool(communicator_independent_recv_thread, true,
            "use an independent to recv vars from parameter server");
DEFINE_int32(communicator_send_queue_size, 20,
             "queue size to recv gradient before send");
DEFINE_int32(communicator_min_send_grad_num_before_recv, 20,
             "max grad num to send before recv parameters");
DEFINE_int32(communicator_thread_pool_size, 5, "thread num to do send or recv");
DEFINE_int32(communicator_send_wait_times, 5,
             "times that send thread will wait if merge num does not reach "
             "max_merge_var_num");
DEFINE_int32(communicator_max_merge_var_num, 20,
             "max var num to merge and send");
DEFINE_bool(communicator_fake_rpc, false,
            "fake mode does not really send any thing");
DEFINE_bool(communicator_merge_sparse_grad, true,
            "merge sparse gradient before sending");

namespace paddle {
namespace operators {
namespace distributed {

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

std::shared_ptr<Communicator> Communicator::communicator_(nullptr);

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
  VLOG(0) << "communicator_min_send_grad_num_before_recv: "
          << FLAGS_communicator_min_send_grad_num_before_recv;
  VLOG(0) << "communicator_thread_pool_size: "
          << FLAGS_communicator_thread_pool_size;
  VLOG(0) << "communicator_send_wait_times: "
          << FLAGS_communicator_send_wait_times;
  VLOG(0) << "communicator_max_merge_var_num: "
          << FLAGS_communicator_max_merge_var_num;
  VLOG(0) << "communicator_fake_rpc: " << FLAGS_communicator_fake_rpc;
  VLOG(0) << "communicator_merge_sparse_grad: "
          << FLAGS_communicator_merge_sparse_grad;
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
  if (FLAGS_v >= 3) {
    std::string msg("~Communicator");
    fwrite(msg.c_str(), msg.length(), 1, stdout);
  }
  running_ = false;
  if (send_thread_) send_thread_->join();
  if (recv_thread_) recv_thread_->join();
  if (FLAGS_v >= 3) {
    std::string msg("~Communicator done");
    fwrite(msg.c_str(), msg.length(), 1, stdout);
  }
}

void Communicator::SendThread() {
  VLOG(3) << "SendThread start!";

  std::unordered_map<std::string, std::vector<std::shared_ptr<Variable>>>
      unmerged_vars;

  for (auto &iter : send_varname_to_ctx_) {
    std::vector<std::shared_ptr<Variable>> vars;
    vars.reserve(FLAGS_communicator_max_merge_var_num);
    unmerged_vars[iter.first] = vars;
  }

  auto &first_var_q = send_varname_to_queue_.begin()->second;
  auto &first_var_v = unmerged_vars.begin()->second;
  auto wait_times = 0;

  while (running_) {
    while (first_var_v.size() < FLAGS_communicator_max_merge_var_num) {
      if (first_var_q->Size() == 0) {
        VLOG(3) << "wait_times -> " << wait_times;
        if (wait_times >= FLAGS_communicator_send_wait_times) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_times++;
        continue;
      } else {
        wait_times = 0;

        for (auto &iter : send_varname_to_queue_) {
          auto &var_name = iter.first;
          auto &var_queue = iter.second;
          auto &var_vec = unmerged_vars.at(var_name);

          var_vec.push_back(var_queue->Pop());
        }
        grad_num_.fetch_add(1, std::memory_order_relaxed);
      }
    }

    if (first_var_v.size() > 0) {
      std::vector<std::future<void>> task_futures;
      task_futures.reserve(send_varname_to_ctx_.size());
      VLOG(3) << "run merge and send graph";

      for (auto &iter : unmerged_vars) {
        auto &var_name = iter.first;
        auto &vars = iter.second;

        auto send_task = [this, &var_name, &vars] {
          auto before_merge = GetCurrentUS();
          MergeVars(var_name, vars, send_scope_.get());
          auto after_merge = GetCurrentUS();

          auto send_functor = distributed::ParameterSend<float>();
          auto &ctx = send_varname_to_ctx_.at(var_name);
          if (!FLAGS_communicator_fake_rpc) {
            send_functor(ctx, *send_scope_, true);
          }
          auto after_send = GetCurrentUS();

          VLOG(3) << var_name << " merge use time "
                  << after_merge - before_merge << " send use time "
                  << after_send - after_merge;
        };
        task_futures.emplace_back(
            send_threadpool_->enqueue(std::move(send_task)));
      }

      for (auto &task_f : task_futures) {
        task_f.wait();
      }

      if (!FLAGS_communicator_independent_recv_thread) {
        RecvAll();
      }

      for (auto &unmerged_var : unmerged_vars) {
        unmerged_var.second.clear();
      }
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      VLOG(3) << "0 vars in send queue, wait";
    }
  }

  for (auto &unmerged_var : unmerged_vars) {
    unmerged_var.second.clear();
  }

  VLOG(0) << "communicator stopped, send thread exit";
}

void Communicator::RecvAll() {
  VLOG(3) << "parallel run recv graph";
  if (!running_) return;
  auto before_send = GetCurrentUS();
  std::vector<std::future<void>> task_futures;
  task_futures.reserve(recv_varname_to_ctx_.size());
  for (auto &iter : recv_varname_to_ctx_) {
    auto recv_task = [this, &iter] {
      auto &var_name = iter.first;
      VLOG(4) << "recv var " << var_name;
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
  auto after_recv = GetCurrentUS();
  VLOG(1) << "run recv graph use time " << after_recv - before_send;
}

void Communicator::RecvThread() {
  VLOG(3) << "RecvThread start!";
  while (running_) {
    auto grad_num = grad_num_.load();
    if (grad_num > FLAGS_communicator_min_send_grad_num_before_recv) {
      VLOG(1) << "current grad num " << grad_num;
      RecvAll();
      grad_num_.store(0);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  VLOG(0) << "communicator stopped, recv thread exit";
}

void Communicator::Send(const std::string &var_name,
                        const framework::Scope &scope) {
  VLOG(3) << "communicator send " << var_name;
  // push var into send queue by var_name
  auto *grad_var = scope.FindVar(var_name);
  PADDLE_ENFORCE(grad_var->IsInitialized(), "grad var should be inited");
  if (grad_var->IsType<framework::SelectedRows>() &&
      !FLAGS_communicator_merge_sparse_grad) {
    auto send_functor = distributed::ParameterSend<float>();
    auto &ctx = send_varname_to_ctx_.at(var_name);
    if (!FLAGS_communicator_fake_rpc) {
      send_functor(ctx, scope, true);
    }
  } else {
    auto tmp_grad_var = std::make_shared<Variable>();
    framework::CopyVariable(*grad_var, tmp_grad_var.get());
    auto &queue = send_varname_to_queue_.at(var_name);
    VLOG(3) << "send " << var_name << " queue size " << queue->Size();
    queue->Push(tmp_grad_var);
  }
}

void Communicator::Init(const paddle::framework::ProgramDesc &program,
                        Scope *param_scope) {
  using RpcCtxMap = operators::distributed::RpcCtxMap;
  VLOG(3) << "ProcessGraph";
  RpcCtxMap send_varname_to_ctx;
  RpcCtxMap recv_varname_to_ctx;
  for (auto *op : program.Block(0).AllOps()) {
    VLOG(3) << "node name " << op->Type();
    if (op->Type() == "send") {
      auto send_var_name = op->Input("X")[0];
      auto send_varnames = boost::get<std::vector<std::string>>(
          op->GetNullableAttr("send_varnames"));
      auto epmap =
          boost::get<std::vector<std::string>>(op->GetNullableAttr("epmap"));
      auto height_section =
          boost::get<std::vector<int64_t>>(op->GetNullableAttr("sections"));
      auto trainer_id = boost::get<int>(op->GetNullableAttr("trainer_id"));
      send_varname_to_ctx[send_var_name] = operators::distributed::RpcContext(
          send_var_name, send_varnames, epmap, height_section, trainer_id);
      VLOG(3) << "find and init an send op: "
              << send_varname_to_ctx[send_var_name];
    } else if (op->Type() == "recv") {
      auto do_not_run = boost::get<int>(op->GetNullableAttr("do_not_run"));
      PADDLE_ENFORCE_GT(do_not_run, 0, "recv should not run!");
      auto recv_var_name = op->Output("Out")[0];
      auto recv_varnames = boost::get<std::vector<std::string>>(
          op->GetNullableAttr("recv_varnames"));
      auto epmap =
          boost::get<std::vector<std::string>>(op->GetNullableAttr("epmap"));
      auto trainer_id = boost::get<int>(op->GetNullableAttr("trainer_id"));
      recv_varname_to_ctx[recv_var_name] = operators::distributed::RpcContext(
          recv_var_name, recv_varnames, epmap, {}, trainer_id);
    }
  }

  // init communicator here
  if (send_varname_to_ctx.size() == 0 && recv_varname_to_ctx.size() == 0) {
    LOG(WARNING) << "no var need to send and recv!!";
  }
  operators::distributed::Communicator::Init(send_varname_to_ctx,
                                             recv_varname_to_ctx, param_scope);
}

Communicator *Communicator::GetInstance() { return communicator_.get(); }

std::shared_ptr<Communicator> Communicator::GetInstantcePtr() {
  return communicator_;
}

void Communicator::Start() {
  VLOG(0) << "Communicator start";
  if (!communicator_) {
    VLOG(0) << "Communicator is not inited, do nothing";
  } else {
    VLOG(1) << "start send thread and recv thread";
    running_ = true;
    // start send and recv thread
    send_thread_.reset(
        new std::thread(std::bind(&Communicator::SendThread, this)));
    if (FLAGS_communicator_independent_recv_thread) {
      recv_thread_.reset(
          new std::thread(std::bind(&Communicator::RecvThread, this)));
    }
  }
}

void Communicator::Stop() {
  VLOG(0) << "Communicator stop";
  running_ = false;
  if (!communicator_) {
    VLOG(0) << "Communicator is not inited, do nothing";
  } else {
    if (send_thread_) {
      VLOG(1) << "stop send thread";
      send_thread_->join();
      send_thread_.reset(nullptr);
    }
    if (recv_thread_) {
      VLOG(1) << "stop recv thread";
      recv_thread_->join();
      recv_thread_.reset(nullptr);
    }
  }
  VLOG(0) << "Communicator stop done";
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
