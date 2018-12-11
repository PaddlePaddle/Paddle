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

#include "paddle/fluid/framework/details/computation_op_handle.h"

#include <string>

namespace paddle {
namespace framework {
namespace details {
ComputationOpHandle::ComputationOpHandle(ir::Node *node, Scope *scope,
                                         platform::Place place)
    : OpHandleBase(node),
      op_(framework::OpRegistry::CreateOp(*node->Op())),
      scope_(scope),
      place_(place) {}

struct RecordTime {
  RecordTime(const std::string &name, const std::string &type)
      : name_(name), type_(type), start_(std::chrono::system_clock::now()) {}

  ~RecordTime() {
    if (type_ == "elementsize_add") {
      end_ = std::chrono::system_clock::now();
      std::chrono::duration<double> diff = end_ - start_;
      VLOG(1) << name_ << " " << type_ << " time record: " << diff.count();
    }
  }

  std::string name_;
  std::string type_;
  std::chrono::system_clock::time_point start_;
  std::chrono::system_clock::time_point end_;
};

void ComputationOpHandle::RunImpl() {
  {
    RecordTime rt("ComputationOpHandle::RunImpl", "Wait");
    WaitInputVarGenerated(place_);
  }

  Scope *scope = nullptr;
  {
    RecordTime rt("ComputationOpHandle::RunImpl", "PrepareScope");
    scope = scope_->FindVar(kLocalExecScopeName)->Get<Scope *>();
  }

  {
    RecordTime rt("ComputationOpHandle::RunImpl", "ReallyRun " + op_->Type());

    auto run_func = [this, scope]() { op_->Run(*scope, place_); };

    if (is_lock_and_record_event_free_) {
      run_func();
    } else {
      this->RunAndRecordEvent(run_func);
    }
  }
}

bool ComputationOpHandle::NeedWait(VarHandleBase *in_var) {
  bool need_wait =
      in_var && in_var->GeneratedOp() &&
      in_var->GeneratedOp()->DeviceContext(place_) != dev_ctxes_.at(place_);
  return need_wait;
}

std::string ComputationOpHandle::Name() const { return op_->Type(); }
}  // namespace details
}  // namespace framework
}  // namespace paddle
