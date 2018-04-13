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

#include "paddle/fluid/framework/details/broadcast_op_handle.h"

namespace paddle {
namespace framework {
namespace details {

Tensor *GetTensorFromVar(Variable *in_var) {
  if (in_var->IsType<LoDTensor>()) {
    return in_var->GetMutable<LoDTensor>();
  } else if (in_var->IsType<SelectedRows>()) {
    return in_var->GetMutable<SelectedRows>()->mutable_value();
  } else {
    PADDLE_THROW("Var should be LoDTensor or SelectedRows");
  }
  return nullptr;
}

BroadcastOpHandle::BroadcastOpHandle(const std::vector<Scope *> &local_scopes,
                                     const std::vector<platform::Place> &places)
    : local_scopes_(local_scopes), places_(places) {}

void BroadcastOpHandle::RunImpl() {
  // the input may have dummy var.
  std::vector<VarHandle *> in_var_handle;
  for (auto *in : inputs_) {
    auto *out_handle = dynamic_cast<VarHandle *>(in);
    if (out_handle) {
      in_var_handle.push_back(out_handle);
    }
  }
  PADDLE_ENFORCE_EQ(in_var_handle.size(), 1,
                    "The number of input should be one.");

  // the output may have dummy var.
  std::vector<VarHandle *> out_var_handles;
  for (auto *out : outputs_) {
    auto *out_handle = dynamic_cast<VarHandle *>(out);
    if (out_handle) {
      out_var_handles.push_back(out_handle);
    }
  }

  PADDLE_ENFORCE_EQ(
      out_var_handles.size(), places_.size(),
      "The number of output should equal to the number of places.");

  // Wait input done, this Wait is asynchronous operation
  auto &in_place = in_var_handle[0]->place_;
  if (in_var_handle[0]->generated_op_) {
    in_var_handle[0]->generated_op_->Wait(dev_ctxes_[in_place]);
    for (auto *out : out_var_handles) {
      auto &out_p = out->place_;
      if (platform::is_same_place(in_place, out_p)) continue;
      in_var_handle[0]->generated_op_->Wait(dev_ctxes_[out_p]);
    }
  }

  //
  auto in_scope_idx = in_var_handle[0]->scope_idx_;
  PADDLE_ENFORCE_LT(in_scope_idx, local_scopes_.size(),
                    "The input(%s) is not in the local_scopes.",
                    in_var_handle[0]->name_);
  auto in_var = local_scopes_[in_scope_idx]->FindVar(in_var_handle[0]->name_);
  Tensor *in_tensor = GetTensorFromVar(in_var);

  for (auto *out : out_var_handles) {
    auto &out_p = out->place_;

    auto out_scope_idx = out->scope_idx_;
    PADDLE_ENFORCE_LT(out_scope_idx, local_scopes_.size(),
                      "%s is not in the local_scopes ", out->name_);

    auto *s = local_scopes_[out_scope_idx];
    auto out_var = s->FindVar(out->name_);
    PADDLE_ENFORCE_EQ(out_p.which(), in_place.which(),
                      "The place of input and output should be the same.");

    if (in_var->IsType<framework::SelectedRows>()) {
      auto &in_sr = in_var->Get<framework::SelectedRows>();
      auto out_sr = out_var->GetMutable<framework::SelectedRows>();
      if (&in_sr == out_sr) continue;
      out_sr->set_height(in_sr.height());
      out_sr->set_rows(in_sr.rows());
      out_sr->mutable_value()->Resize(in_sr.value().dims());
      out_sr->mutable_value()->mutable_data(out_p, in_sr.value().type());
    } else if (in_var->IsType<framework::LoDTensor>()) {
      auto in_lod = in_var->Get<framework::LoDTensor>();
      auto out_lod = out_var->GetMutable<framework::LoDTensor>();
      if (&in_lod == out_lod) continue;
      out_lod->set_lod(in_lod.lod());
      out_lod->Resize(in_lod.dims());
      out_lod->mutable_data(out_p, in_lod.type());
    } else {
      PADDLE_THROW("Var should be LoDTensor or SelectedRows.");
    }

    Tensor *out_tensor = GetTensorFromVar(out_var);
    if (platform::is_cpu_place(in_place)) {
      paddle::framework::TensorCopy(*in_tensor, out_p, *(dev_ctxes_[in_place]),
                                    out_tensor);
    } else if (platform::is_gpu_place(in_place)) {
#ifdef PADDLE_WITH_CUDA
      auto src_gpu_place = boost::get<platform::CUDAPlace>(in_place);
      auto dst_gpu_place = boost::get<platform::CUDAPlace>(out_p);
      void *dst_ptr = out_tensor->mutable_data(out_p);
      void *src_ptr = in_tensor->data<void>();
      int64_t size = in_tensor->numel() * SizeOfType(in_tensor->type());
      memory::Copy(
          dst_gpu_place, dst_ptr, src_gpu_place, src_ptr, size,
          reinterpret_cast<platform::CUDADeviceContext *>(dev_ctxes_[out_p])
              ->stream());
#else
      PADDLE_THROW("CUDAPlace is not supported in CPU device.");
#endif
    }
  }
}

std::string BroadcastOpHandle::Name() const { return "broadcast"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
