//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/details/op_handle_base.h"
#include <map>
#include <unordered_set>

namespace paddle {
namespace framework {
namespace details {
std::string OpHandleBase::DebugString() const {
  std::stringstream ss;
  ss << Name() << "(";
  for (auto *var : inputs_) {
    ss << var->DebugString() << ", ";
  }
  ss << ") --> (";
  for (auto *var : outputs_) {
    ss << var->DebugString() << ", ";
  }
  ss << ")\n";
  return ss.str();
}

OpHandleBase::~OpHandleBase() PADDLE_MAY_THROW {
#ifdef PADDLE_WITH_CUDA
  for (auto &ev : events_) {
    if (ev.second) {
      PADDLE_ENFORCE(cudaEventDestroy(ev.second));
    }
  }
#endif
}

void OpHandleBase::InitCUDA() {
#ifdef PADDLE_WITH_CUDA
  for (auto &p : dev_ctxes_) {
    int dev_id = BOOST_GET_CONST(platform::CUDAPlace, p.first).device;
    PADDLE_ENFORCE(cudaSetDevice(dev_id));
    PADDLE_ENFORCE(
        cudaEventCreateWithFlags(&events_[dev_id], cudaEventDisableTiming));
  }
  if (IsMultiDeviceTransfer() && dev_ctxes_.size() > 0) {
    for (auto &out_var : outputs_) {
      auto *out_var_handle = dynamic_cast<VarHandle *>(out_var);
      if (out_var_handle) {
        int dev_id =
            BOOST_GET_CONST(platform::CUDAPlace, out_var_handle->place())
                .device;
        out_var_handle->SetGenerateEvent(events_.at(dev_id));
      }
    }
  } else {
    PADDLE_ENFORCE_EQ(dev_ctxes_.size(), 1UL,
                      "%s should have only one dev_ctx.", Name());
    auto &place = dev_ctxes_.begin()->first;
    int dev_id = BOOST_GET_CONST(platform::CUDAPlace, place).device;
    for (auto &out_var : outputs_) {
      auto *out_var_handle = dynamic_cast<VarHandle *>(out_var);
      if (out_var_handle) {
        PADDLE_ENFORCE(platform::is_same_place(place, out_var_handle->place()),
                       "The place of output(%s) is not consistent with the "
                       "place of current op(%s).",
                       out_var_handle->Name(), Name());
        out_var_handle->SetGenerateEvent(events_.at(dev_id));
      }
    }
  }
#endif
}

void OpHandleBase::Run(bool use_cuda) {
#ifdef PADDLE_WITH_CUDA
  if (events_.empty() && use_cuda && dev_ctxes_.size() > 0) {
    InitCUDA();
  }
#else
  PADDLE_ENFORCE(!use_cuda);
#endif

  RunImpl();
}

void OpHandleBase::RecordWaitEventOnCtx(platform::DeviceContext *waited_ctx) {
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_NOT_NULL(waited_ctx);
  if (platform::is_cpu_place(waited_ctx->GetPlace()) || events_.empty()) {
    for (auto &dev_ctx : dev_ctxes_) {
      PADDLE_ENFORCE_NOT_NULL(dev_ctx.second);
      dev_ctx.second->Wait();
    }
  } else {
    auto stream =
        static_cast<platform::CUDADeviceContext *>(waited_ctx)->stream();
    for (auto &ev : events_) {
      PADDLE_ENFORCE(cudaStreamWaitEvent(stream, ev.second, 0));
    }
  }
#else
  for (auto &dev_ctx : dev_ctxes_) {
    dev_ctx.second->Wait();
  }
#endif
}

void OpHandleBase::AddInput(VarHandleBase *in) {
  this->inputs_.emplace_back(in);
  node_->inputs.push_back(in->Node());
  in->AddOutput(this, this->Node());
}

void OpHandleBase::AddOutput(VarHandleBase *out) {
  outputs_.emplace_back(out);
  node_->outputs.push_back(out->Node());
  out->AddInput(this, this->Node());
}

void OpHandleBase::WaitInputVarGenerated() {
  for (auto in_var : inputs_) {
    if (NeedWait(in_var)) {
      // Dummy Variable is used to represent dependencies between operators, so
      // there doesn't add event for it.
      auto *in_var_handle = dynamic_cast<VarHandle *>(in_var);
      if (in_var_handle) {
        auto &place = in_var_handle->place();
        if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
          auto stream =
              static_cast<platform::CUDADeviceContext *>(dev_ctxes_.at(place))
                  ->stream();
          PADDLE_ENFORCE(
              cudaStreamWaitEvent(stream, in_var_handle->GetEvent(), 0));
#else
          PADDLE_THROW("Doesn't compile the GPU.");
#endif
        }
        // There are nothing to do when the place is CPUPlace.
      }
    }
  }
}

void OpHandleBase::WaitInputVarGenerated(const platform::Place &place) {
  for (auto in_var : inputs_) {
    if (NeedWait(in_var)) {
      // Dummy Variable is used to represent dependencies between operators, so
      // there doesn't add event for it.
      auto *in_var_handle = dynamic_cast<VarHandle *>(in_var);
      if (in_var_handle) {
        if (platform::is_gpu_place(in_var_handle->place())) {
#ifdef PADDLE_WITH_CUDA
          auto stream = static_cast<platform::CUDADeviceContext *>(
                            dev_ctxes_.at(in_var_handle->place()))
                            ->stream();
          PADDLE_ENFORCE(
              cudaStreamWaitEvent(stream, in_var_handle->GetEvent(), 0));
#else
          PADDLE_THROW("Doesn't compile the GPU.");
#endif
        }
        // There are nothing to do when the place is CPUPlace.
      }
    }
  }
}

size_t OpHandleBase::NoDummyInputSize() const {
  size_t cnt = 0;
  for (auto *in : inputs_) {
    if (dynamic_cast<DummyVarHandle *>(in) == nullptr) {
      ++cnt;
    }
  }
  return cnt;
}

bool OpHandleBase::NeedWait(VarHandleBase *in_var) {
  return in_var && in_var->GeneratedOp();
}

void OpHandleBase::RunAndRecordEvent(const std::function<void()> &callback) {
  callback();
#ifdef PADDLE_WITH_CUDA
  if (!events_.empty()) {  // Use event
    for (auto &p : dev_ctxes_) {
      auto dev_id = BOOST_GET_CONST(platform::CUDAPlace, p.first).device;
      auto *cuda_dev_ctx = static_cast<platform::CUDADeviceContext *>(p.second);
      VLOG(10) << "cudadevicecontext:" << cuda_dev_ctx << ", dev_id:" << dev_id;
      PADDLE_ENFORCE_CUDA_SUCCESS(
          cudaEventRecord(events_.at(dev_id), cuda_dev_ctx->stream()));
    }
  }
#endif
}

void OpHandleBase::RunAndRecordEvent(platform::Place p,
                                     const std::function<void()> &callback) {
#ifdef PADDLE_WITH_CUDA
  if (platform::is_cpu_place(p) || events_.empty()) {
    callback();
  } else {
    auto *ctx = dev_ctxes_.at(p);
    auto *cuda_ctx = static_cast<platform::CUDADeviceContext *>(ctx);
    cuda_ctx->RecordEvent(
        events_.at(BOOST_GET_CONST(platform::CUDAPlace, p).device), callback);
  }
#else
  callback();
#endif
}

size_t OpHandleBase::NotReadyInputSize() const {
  std::unordered_set<VarHandleBase *> res;
  for (auto *var : inputs_) {
    if (var->GeneratedOp() != nullptr) {
      res.emplace(var);
    }
  }
  return res.size();
}

void OpHandleBase::SetLocalExecScopes(
    const std::unordered_map<Scope *, Scope *> &scope_map) {
  local_exec_scopes_.clear();
  auto scopes = GetLocalScopes();
  for (auto *scope : scopes) {
    auto iter = scope_map.find(scope);
    PADDLE_ENFORCE(iter != scope_map.end(), "Local scope not found");
    local_exec_scopes_.emplace_back(iter->second);
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
