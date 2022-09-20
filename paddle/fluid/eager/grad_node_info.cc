// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/grad_node_info.h"

#include "glog/logging.h"
#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

/**
 * Implementation of GradNodeBase, Edge and GradTensorHolder.
 **/
namespace egr {

static void CheckTensor(const paddle::experimental::Tensor& pre,
                        const paddle::experimental::Tensor& post) {
  if (!pre.initialized() && post.initialized()) {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "The tensor in before and after hook are not consistent"));
  }
  if (pre.initialized() && post.initialized()) {
    VLOG(7) << paddle::framework::DataType2String(pre.dtype()) << " "
            << paddle::framework::DataType2String(post.dtype());
    PADDLE_ENFORCE_EQ(
        pre.dtype(),
        post.dtype(),
        paddle::platform::errors::PermissionDenied(
            "The dtype of tensor before(%s) and after(%s) hook are not "
            "consistent",
            paddle::framework::DataType2String(pre.dtype()),
            paddle::framework::DataType2String(post.dtype())));
    PADDLE_ENFORCE_EQ(pre.place(),
                      post.place(),
                      paddle::platform::errors::PermissionDenied(
                          "The place of tensor before(%s) and after(%s) "
                          "hook are not consistent",
                          pre.place().DebugString(),
                          post.place().DebugString()));
  }
}

GradNodeBase::GradNodeBase(size_t bwd_in_slot_num, size_t bwd_out_slot_num) {
  VLOG(7) << "Construct GradNodeBase";
  bwd_in_meta_.resize(bwd_in_slot_num);
  bwd_out_meta_.resize(bwd_out_slot_num);
}

const paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
GradNodeBase::InputMeta() const {
  return bwd_in_meta_;
}

const paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
GradNodeBase::OutputMeta() const {
  return bwd_out_meta_;
}

paddle::small_vector<std::vector<GradSlotMeta>, kSlotSmallVectorSize>&
GradNodeBase::MutableOutputMeta() {
  return bwd_out_meta_;
}

void GradNodeBase::SetGradInMeta(const paddle::experimental::Tensor& fwd_out,
                                 size_t slot_rank) {
  VLOG(7) << "Set GradSlotMeta for Grad Inputs";
  auto* fwd_out_meta = egr::EagerUtils::nullable_autograd_meta(fwd_out);
  PADDLE_ENFORCE_LE(
      slot_rank,
      (bwd_in_meta_.size() - 1),
      paddle::platform::errors::InvalidArgument(
          "Slot Rank should less equal than bwd_in_meta_ size, since "
          "bwd_in_meta_ is designed to hold as same num as backward "
          "inputs."));
  auto& metas = bwd_in_meta_.at(slot_rank);
  if (metas.size() == 0) {
    metas.resize(1);
  }

  auto& meta = metas[0];
  if (fwd_out_meta && fwd_out_meta->StopGradient()) {
    meta.SetStopGradient(fwd_out_meta->StopGradient());
  }

  if (!fwd_out.initialized()) {
    VLOG(7)
        << "Skip Configuring GradSlotMeta for uninitialized GradInput Tensor";
    return;
  }

  phi::DenseTensor* dense_tensor = nullptr;
  // Record TensorMeta
  if (phi::DenseTensor::classof(fwd_out.impl().get())) {
    // Only Copy Meta
    dense_tensor = static_cast<phi::DenseTensor*>(fwd_out.impl().get());
  } else if (phi::SparseCooTensor::classof(fwd_out.impl().get())) {
    phi::SparseCooTensor* coo_tensor =
        static_cast<phi::SparseCooTensor*>(fwd_out.impl().get());
    dense_tensor = coo_tensor->mutable_non_zero_elements();
  } else if (phi::SparseCsrTensor::classof(fwd_out.impl().get())) {
    phi::SparseCsrTensor* csr_tensor =
        static_cast<phi::SparseCsrTensor*>(fwd_out.impl().get());
    dense_tensor = csr_tensor->mutable_non_zero_elements();
  } else {
    VLOG(7) << "Unable to initialize the DenseTensorMeta of GradSlotMeta with "
               "non-DenseTensor argument.";
  }
  PADDLE_ENFORCE_NE(
      dense_tensor->meta().dtype,
      phi::DataType::UNDEFINED,
      paddle::platform::errors::Fatal(
          "Attempting to copy DenseTensorMeta with phi::DataType::UNDEFINED,"
          "which is illegal."));

  meta.SetTensorMeta(dense_tensor->meta());
  meta.SetPlace(fwd_out.place());

  if (dense_tensor->type() == paddle::experimental::DataType::COMPLEX64 ||
      dense_tensor->type() == paddle::experimental::DataType::COMPLEX128) {
    need_complex_to_real_ = true;
  }
}

void GradNodeBase::SetGradInMeta(
    const std::vector<paddle::experimental::Tensor>& fwd_out,
    size_t slot_rank) {
  VLOG(7) << "Set GradSlotMeta for Grad Inputs";
  size_t slot_size = fwd_out.size();
  PADDLE_ENFORCE_LE(
      slot_rank,
      (bwd_in_meta_.size() - 1),
      paddle::platform::errors::InvalidArgument(
          "Slot Rank should less equal than bwd_in_meta_ size, since "
          "bwd_in_meta_ is designed to hold as same num as backward "
          "inputs."));
  auto& metas = bwd_in_meta_.at(slot_rank);
  // Init stop gradient vector before use to avoid push back
  if (metas.size() < slot_size) {
    VLOG(7) << "Init bwd_in_meta_ with slot rank: " << slot_rank;
    metas.resize(slot_size);
  }
  for (size_t i = 0; i < slot_size; i++) {
    auto& meta = metas[i];
    const auto& fwd_out_tensor = fwd_out[i];
    auto* fwd_out_meta =
        egr::EagerUtils::nullable_autograd_meta(fwd_out_tensor);
    PADDLE_ENFORCE_NOT_NULL(fwd_out_meta,
                            paddle::platform::errors::PreconditionNotMet(
                                "Bwd_in_meta should only be called while "
                                "autograd_meta is not null. If you got this "
                                "error, it indicates bugs in framework."));
    if (fwd_out_meta && fwd_out_meta->StopGradient()) {
      // Set Stop Gradient only when its true or non-initialized autograd_meta,
      // since all default value is false.
      meta.SetStopGradient(fwd_out_meta->StopGradient());
    }

    if (!fwd_out_tensor.initialized()) {
      VLOG(7)
          << "Skip Configuring GradSlotMeta for uninitialized GradInput Tensor";
      return;
    }

    // Record TensorMeta
    if (phi::DenseTensor::classof(fwd_out_tensor.impl().get())) {
      // Only Copy Meta
      phi::DenseTensor* dense_tensor =
          static_cast<phi::DenseTensor*>(fwd_out_tensor.impl().get());

      PADDLE_ENFORCE_NE(
          dense_tensor->meta().dtype,
          phi::DataType::UNDEFINED,
          paddle::platform::errors::Fatal("Attempting to copy DenseTensorMeta "
                                          "with phi::DataType::UNDEFINED,"
                                          "which is illegal."));
      meta.SetTensorMeta(dense_tensor->meta());
      meta.SetPlace(fwd_out_tensor.place());

      if (dense_tensor->type() == paddle::experimental::DataType::COMPLEX64 ||
          dense_tensor->type() == paddle::experimental::DataType::COMPLEX128) {
        need_complex_to_real_ = true;
      }
    } else {
      VLOG(7) << "Unable to initialize the DenseTensorMeta of GradSlotMeta "
                 "with non-DenseTensor argument.";
    }
  }
}

void GradNodeBase::SetGradOutMeta(const paddle::experimental::Tensor& fwd_in,
                                  size_t slot_rank) {
  auto* fwd_in_meta = egr::EagerUtils::nullable_autograd_meta(fwd_in);
  PADDLE_ENFORCE_LE(
      (slot_rank + 1),
      bwd_out_meta_.size(),
      paddle::platform::errors::InvalidArgument(
          "Slot Rank should less equal than bwd_out_meta_ size, "
          "since bwd_out_meta_ is designed to hold as same num as "
          "backward outputs."));
  auto& metas = bwd_out_meta_.at(slot_rank);
  // Init stop gradient vector before use to avoid push back
  if (metas.size() == 0) {
    metas.resize(1);
  }
  auto& meta = metas[0];
  // Set Stop_gradient
  if (fwd_in_meta) {
    meta.SetStopGradient(fwd_in_meta->StopGradient());
  } else {
    meta.SetStopGradient(true);
  }
  // Set Adj Edges
  if (fwd_in_meta && !fwd_in_meta->StopGradient()) {
    auto node = fwd_in_meta->GetMutableGradNode();
    if (!node || !node.get()) {
      fwd_in_meta->SetGradNode(
          std::make_shared<egr::GradNodeAccumulation>(fwd_in_meta));
    }
    VLOG(3) << "Add Edges for slot: " << slot_rank << ", the Edge is from "
            << this->name() << " (addr: " << this << ") "
            << " to " << fwd_in_meta->GetMutableGradNode()->name()
            << " (addr: " << fwd_in_meta->GetMutableGradNode().get() << ")";

    meta.SetEdge(fwd_in_meta->GetMutableGradNode(), fwd_in_meta->OutRankInfo());
  }
  // Record TensorMeta
  if (fwd_in.impl() && fwd_in.impl().get()) {
    if (phi::DenseTensor::classof(fwd_in.impl().get())) {
      // Only Copy Meta
      phi::DenseTensor* dense_tensor =
          static_cast<phi::DenseTensor*>(fwd_in.impl().get());
      PADDLE_ENFORCE_NE(
          dense_tensor->meta().dtype,
          phi::DataType::UNDEFINED,
          paddle::platform::errors::Fatal("Attempting to copy DenseTensorMeta "
                                          "with phi::DataType::UNDEFINED,"
                                          "which is illegal."));
      meta.SetTensorMeta(dense_tensor->meta());
      meta.SetPlace(fwd_in.place());
    }
  } else {
    VLOG(7) << "Unable to initialize the DenseTensorMeta of GradSlotMeta with "
               "non-DenseTensor argument.";
  }
}

void GradNodeBase::SetGradOutMeta(
    const std::vector<paddle::experimental::Tensor>& fwd_in, size_t slot_rank) {
  size_t slot_size = fwd_in.size();
  PADDLE_ENFORCE_LE(
      slot_rank,
      (bwd_out_meta_.size() - 1),
      paddle::platform::errors::InvalidArgument(
          "Slot Rank should less equal than bwd_out_meta_ size, "
          "since bwd_out_meta_ is designed to hold as same num as "
          "backward outputs."));
  auto& metas = bwd_out_meta_.at(slot_rank);
  // Init stop gradient vector before use to avoid push back
  if (metas.size() < slot_size) {
    metas.resize(slot_size);
  }
  for (size_t i = 0; i < slot_size; i++) {
    const auto& fwd_in_tensor = fwd_in[i];
    auto& meta = metas[i];
    auto* fwd_in_meta = egr::EagerUtils::nullable_autograd_meta(fwd_in_tensor);
    // Set Stop_gradient
    if (fwd_in_meta) {
      meta.SetStopGradient(fwd_in_meta->StopGradient());
    }
    // Set Adj Edges
    if (fwd_in_meta && !fwd_in_meta->StopGradient()) {
      auto node = fwd_in_meta->GetMutableGradNode();
      if (!node || !node.get()) {
        fwd_in_meta->SetGradNode(
            std::make_shared<egr::GradNodeAccumulation>(fwd_in_meta));
      }
      VLOG(3) << "Add Edges for slot: " << slot_rank << ", the Edge is from "
              << this->name() << " (addr: " << this << ") "
              << " to " << fwd_in_meta->GetMutableGradNode()->name()
              << " (addr: " << fwd_in_meta->GetMutableGradNode().get() << ")";

      meta.SetEdge(fwd_in_meta->GetMutableGradNode(),
                   fwd_in_meta->OutRankInfo());
    }
    // Record TensorMeta
    if (fwd_in_tensor.impl() && fwd_in_tensor.impl().get()) {
      if (phi::DenseTensor::classof(fwd_in_tensor.impl().get())) {
        // Only Copy Meta
        phi::DenseTensor* dense_tensor =
            static_cast<phi::DenseTensor*>(fwd_in_tensor.impl().get());
        PADDLE_ENFORCE_NE(dense_tensor->dtype(),
                          phi::DataType::UNDEFINED,
                          paddle::platform::errors::Fatal(
                              "Attempting to copy DenseTensorMeta "
                              "with phi::DataType::UNDEFINED,"
                              "which is illegal."));
        meta.SetTensorMeta(dense_tensor->meta());
        meta.SetPlace(fwd_in_tensor.place());
      }
    } else {
      VLOG(7)
          << "Unable to initialize the DenseTensorMeta of GradSlotMeta with "
             "non-DenseTensor argument.";
    }
  }
}

void GradNodeBase::SetDefaultGradInOutMeta() {
  PADDLE_ENFORCE((bwd_out_meta_.size() == 1) && (bwd_in_meta_.size() == 1),
                 paddle::platform::errors::PreconditionNotMet(
                     "We can only support 1 input and 1 output in default grad "
                     "meta setter, other size of inputs and outputs should "
                     "create with Setter and Getters"));
  // Default stop_gradient is false and slot id is 0, slot size is 1;
  bwd_out_meta_[0].resize(1);
  bwd_in_meta_[0].resize(1);
}

int64_t GradNodeBase::RegisterGradientHook(
    size_t slot_id, size_t rank, std::shared_ptr<egr::TensorHook>&& hook) {
  gradient_hooks_.emplace(next_hook_id_,
                          std::make_tuple(slot_id, rank, std::move(hook)));
  return next_hook_id_++;
}

paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                     kSlotSmallVectorSize>
GradNodeBase::ApplyGradientHooks(
    const paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                               kSlotSmallVectorSize>& tensors) {
  paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                       kSlotSmallVectorSize>
      outs(tensors.size());
  for (auto& hook_pair : gradient_hooks_) {
    size_t slot_id = std::get<0>(hook_pair.second);
    size_t rank = std::get<1>(hook_pair.second);

    auto hook = std::get<2>(hook_pair.second);

    PADDLE_ENFORCE(slot_id < tensors.size(),
                   paddle::platform::errors::Fatal(
                       "Slot_id from registered hook should be smaller than "
                       "slot size of grad_tensors"));

    PADDLE_ENFORCE(rank < tensors[slot_id].size(),
                   paddle::platform::errors::Fatal(
                       "rank of slot %d from registered hook should be smaller "
                       "than rank size of grad_tensors",
                       slot_id));

    std::vector<paddle::experimental::Tensor>& slot_out = outs[slot_id];
    slot_out.resize(tensors[slot_id].size());
    paddle::experimental::Tensor& out = slot_out[rank];
    if (!out.defined() || !out.initialized()) {
      out = (*hook)(tensors[slot_id][rank]);
    } else {
      // If more than one hook is registered, the input to the next hook func
      // should be the output of the previous hook
      out = (*hook)(out);
    }
  }

  for (size_t i = 0; i < outs.size(); i++) {
    if (outs[i].empty() && (!tensors[i].empty())) {
      outs[i].resize(tensors[i].size());
    }
    // TODO(Jiabin): Optimize this if we only add hook slot by slot
    for (size_t j = 0; j < outs[i].size(); j++) {
      if (!outs[i][j].defined() || !outs[i][j].initialized()) {
        outs[i][j] = tensors[i][j];
      }
      CheckTensor(tensors[i][j], outs[i][j]);
    }
  }

  return outs;
}

void GradNodeBase::HandleComplexGradToRealGrad(
    paddle::small_vector<std::vector<paddle::experimental::Tensor>,
                         kSlotSmallVectorSize>* out_grads) {
  for (size_t slot_id = 0; slot_id < out_grads->size(); slot_id++) {
    const std::vector<paddle::experimental::Tensor>& slot_out_grads =
        (*out_grads)[slot_id];
    for (size_t rank_id = 0; rank_id < slot_out_grads.size(); rank_id++) {
      const GradSlotMeta& slot_meta = bwd_out_meta_[slot_id][rank_id];

      PADDLE_ENFORCE(
          slot_meta.HasTensorMeta() > 0,
          paddle::platform::errors::Fatal(
              "We require TensorMeta in GradInputMeta() to obtain forward data "
              "types."
              "However, no TensorMeta is detected in bwd_out_meta_."));

      auto fwd_data_type = paddle::framework::TransToProtoVarType(
          slot_meta.GetTensorMeta().dtype);
      const paddle::experimental::Tensor& grad = slot_out_grads[rank_id];

      if (paddle::framework::IsComplexType(fwd_data_type)) continue;

      // Only Handle Complex To Real for DenseTensor for now
      if (phi::DenseTensor::classof(grad.impl().get())) {
        phi::DenseTensor* grad_dense_tensor =
            static_cast<phi::DenseTensor*>(grad.impl().get());

        auto curr_data_type =
            paddle::framework::TransToProtoVarType(grad_dense_tensor->type());
        if (!paddle::framework::IsComplexType(curr_data_type)) continue;

        // Convert Complex GradOut to Real
        auto out = std::make_shared<phi::DenseTensor>();
        paddle::framework::TransComplexToReal(
            fwd_data_type, curr_data_type, *grad_dense_tensor, out.get());

        (*out_grads)[slot_id][rank_id].set_impl(out);
      }
    }
  }
}

}  // namespace egr
