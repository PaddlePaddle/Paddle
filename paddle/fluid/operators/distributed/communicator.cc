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

#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/distributed/parameter_recv.h"
#include "paddle/fluid/operators/distributed/parameter_send.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {
namespace distributed {

static inline void MergeVars(const std::string &var_name,
                             const std::vector<std::shared_ptr<Variable>> &vars,
                             Scope *scope) {
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

void Communicator::SendThread() {
  while (running_) {
    std::vector<std::future<void>> task_futures;
    task_futures.reserve(send_varname_to_ctx_.size());
    for (auto &iter : send_varname_to_queue_) {
      auto send_task = [this, &iter] {
        auto &var_name = iter.first;
        VLOG(3) << "merge var " << var_name << " and send";
        auto &var_queue = iter.second;
        std::vector<std::shared_ptr<Variable>> vars;
        // TODO(qiao): need to be configurable
        const size_t max_merge_var_num = 20;
        size_t merged_var_num = 0;
        while (var_queue->Size() > 0 && merged_var_num < max_merge_var_num) {
          vars.push_back(var_queue->Pop());
          merged_var_num++;
        }
        MergeVars(var_name, vars, send_scope_.get());
        auto send_functor = distributed::ParameterSend<float>();
        auto &ctx = send_varname_to_ctx_.at(var_name);
        send_functor(ctx, *send_scope_, true);
      };
      task_futures.emplace_back(
          send_threadpool_->enqueue(std::move(send_task)));
    }
    for (auto &task_f : task_futures) {
      task_f.wait();
    }
  }
}

void Communicator::RecvThread() {
  while (running_) {
    // parallel run recv graph
    std::vector<std::future<void>> task_futures;
    task_futures.reserve(recv_varname_to_ctx_.size());
    for (auto &iter : recv_varname_to_ctx_) {
      auto recv_task = [this, &iter] {
        auto &var_name = iter.first;
        VLOG(3) << "recv var " << var_name;
        auto recv_functor = distributed::ParameterRecv<float>();
        recv_functor(iter.second, *recv_scope_);
      };
      task_futures.emplace_back(
          recv_threadpool_->enqueue(std::move(recv_task)));
    }
    for (auto &task : task_futures) {
      task.wait();
    }
  }
}

void Communicator::Send(const std::string &var_name,
                        const framework::Scope &scope) {
  // push var into send queue by var_name
  auto *grad_var = scope.FindVar(var_name);
  PADDLE_ENFORCE(grad_var->IsInitialized(), "grad var should be inited");
  auto tmp_grad_var = std::make_shared<Variable>();
  framework::CopyVariable(*grad_var, tmp_grad_var.get());
  send_varname_to_queue_[var_name]->Push(tmp_grad_var);
}

void Communicator::Start() {
  running_ = true;
  // start send and recv thread
  send_thread_.reset(
      new std::thread(std::bind(&Communicator::SendThread, this)));
  recv_thread_.reset(
      new std::thread(std::bind(&Communicator::RecvThread, this)));
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
