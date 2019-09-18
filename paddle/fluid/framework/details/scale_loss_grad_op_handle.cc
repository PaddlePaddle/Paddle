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

#include "paddle/fluid/framework/details/scale_loss_grad_op_handle.h"
#include <string>
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace framework {
namespace details {
ScaleLossGradOpHandle::ScaleLossGradOpHandle(ir::Node *node, size_t num_dev,
                                             Scope *scope,
                                             platform::Place place,
                                             platform::DeviceContext *dev_ctx,
                                             proto::VarType::Type dtype)
    : OpHandleBase(node),
      coeff_(static_cast<float>(1.0 / num_dev)),
      scope_(scope),
      place_(place),
      out_dtype_(dtype) {
  this->SetDeviceContext(place_, dev_ctx);
}

ScaleLossGradOpHandle::~ScaleLossGradOpHandle() {}

struct ScaleLossGradFunctor {
  float coeff_;
  Tensor *out_;
  platform::Place place_;
  proto::VarType::Type out_dtype_;
  platform::DeviceContext *ctx_;

  ScaleLossGradFunctor(float coeff, Tensor *out, platform::Place place,
                       proto::VarType::Type dtype, platform::DeviceContext *ctx)
      : coeff_(coeff), out_(out), place_(place), out_dtype_(dtype), ctx_(ctx) {}

  template <typename OutT>
  void apply() const {
    auto *out_data = out_->mutable_data<OutT>(place_);
    if (platform::is_cpu_place(place_)) {
      *out_data = static_cast<OutT>(coeff_);
    } else {
#ifdef PADDLE_WITH_CUDA
      OutT cast_coeff = static_cast<OutT>(coeff_);
      auto stream = static_cast<platform::CUDADeviceContext *>(ctx_)->stream();
      memory::Copy(boost::get<platform::CUDAPlace>(place_), out_data,
                   platform::CPUPlace(), &cast_coeff, SizeOfType(out_dtype_),
                   stream);
      VLOG(10) << place_ << "RUN Scale loss grad op";

#endif
    }
  }
};

void ScaleLossGradOpHandle::RunImpl() {
  platform::RecordEvent record_event(Name());
  // Doesn't wait any event
  std::string var_name = static_cast<VarHandle *>(this->outputs_[0])->name();

  auto *tensor =
      local_exec_scopes_[0]->FindVar(var_name)->GetMutable<LoDTensor>();
  tensor->Resize(make_ddim({1}));

#ifdef PADDLE_WITH_CUDA
  ScaleLossGradFunctor func(coeff_, tensor, place_, out_dtype_,
                            this->dev_ctxes_.at(place_));
  this->RunAndRecordEvent([&] { framework::VisitDataType(out_dtype_, func); });
#else
  ScaleLossGradFunctor func(coeff_, tensor, place_, out_dtype_, nullptr);
  framework::VisitDataType(out_dtype_, func);
#endif
}

std::string ScaleLossGradOpHandle::Name() const { return "Scale LossGrad"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
