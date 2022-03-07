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

#include "paddle/fluid/framework/details/fetch_op_handle.h"

#include <string>

#include "paddle/fluid/platform/profiler/event_tracing.h"

namespace paddle {
namespace framework {
namespace details {

FetchOpHandle::FetchOpHandle(ir::Node *node, FetchResultType *data,
                             size_t offset, std::vector<Scope *> *local_scopes,
                             std::vector<Scope *> *local_exec_scopes,
                             bool return_merged)
    : OpHandleBase(node),
      data_(data),
      offset_(offset),
      local_scopes_(local_scopes),
      local_exec_scopes_(local_exec_scopes),
      return_merged_(return_merged) {}

FetchOpHandle::~FetchOpHandle() {}

void FetchOpHandle::RecordWaitEventOnCtx(platform::DeviceContext *waited_ctx) {
  PADDLE_THROW(platform::errors::PermissionDenied(
      "No nodes need to wait FetchOp. Unexpceted Error."));
}

static void CheckDims(const framework::DDim &tensor_dims,
                      const framework::DDim &ele_dims, const size_t offset) {
  PADDLE_ENFORCE_EQ(
      tensor_dims.size(), ele_dims.size(),
      platform::errors::Fatal("The dimension sizes of fetched Tensors or "
                              "the items of fetched LoDTensorArray are "
                              "different from each other on different "
                              "devices. And the error is caused by the %zu "
                              "(th) fetched variable. Please set the "
                              "parameter `return_merged = False` when you "
                              "call the `Executor.run()` method.",
                              offset));
  for (int j = 1; j < tensor_dims.size(); j++) {
    PADDLE_ENFORCE_EQ(
        tensor_dims[j], ele_dims[j],
        platform::errors::Fatal("The dimensions of fetched Tensors or "
                                "the items of fetched LoDTensorArray are "
                                "different from each other on different "
                                "devices. And the error is caused by the "
                                "%zu (th) fetched variable. Please set the "
                                "parameter `return_merged = False` when "
                                "you call the `Executor.run()` method.",
                                offset));
  }
}

void FetchOpHandle::WaitAndMergeCPUFetchVars() const {
  if (return_merged_) {
    if (data_is_lod_tensor(tensors_[0])) {
      const auto &tensor_dims = BOOST_GET_CONST(LoDTensor, tensors_[0]).dims();
      for (size_t i = 1; i < tensors_.size(); i++) {
        const auto &ele_dims = BOOST_GET_CONST(LoDTensor, tensors_[i]).dims();
        CheckDims(tensor_dims, ele_dims, offset_);
      }
      std::vector<const LoDTensor *> tensors_ptr;
      tensors_ptr.reserve(tensors_.size());
      for (auto &t : tensors_) {
        tensors_ptr.emplace_back(&BOOST_GET_CONST(LoDTensor, t));
      }
      auto &val = BOOST_GET(FetchList, *data_);
      LoDTensor var;
      MergeLoDTensor(&var, tensors_ptr, platform::CPUPlace());
      val.at(offset_) = std::move(var);
    } else {
      auto &array = BOOST_GET_CONST(LoDTensorArray, tensors_[0]);
      LoDTensorArray tmp_array;
      tmp_array.reserve(array.size());
      for (size_t i = 0; i < array.size(); ++i) {
        const auto &tensor_dims = array[i].dims();
        std::vector<const LoDTensor *> tensors_ptr;
        tensors_ptr.reserve(tensors_.size());
        tensors_ptr.push_back(&array[i]);
        for (size_t j = 1; j < tensors_.size(); ++j) {
          auto &element = BOOST_GET_CONST(LoDTensorArray, tensors_[j]);
          const auto &ele_dims = element[i].dims();
          CheckDims(tensor_dims, ele_dims, offset_);
          tensors_ptr.push_back(&element[i]);
        }
        tmp_array.emplace_back();
        MergeLoDTensor(&(tmp_array.back()), tensors_ptr, platform::CPUPlace());
      }
      auto &val = BOOST_GET(FetchList, *data_);
      val.at(offset_) = std::move(tmp_array);
    }
  } else {
    auto &val = BOOST_GET(FetchUnmergedList, *data_);
    val.at(offset_) = std::move(tensors_);
  }
}

static void TransData(const framework::LoDTensor &src_item,
                      framework::LoDTensor *dst_item) {
  if (src_item.IsInitialized() && src_item.numel() > 0) {
    if (platform::is_gpu_place(src_item.place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      TensorCopy(src_item, platform::CPUPlace(), dst_item);
#endif
    } else {
      TensorCopy(src_item, platform::CPUPlace(), dst_item);
    }
  } else {
    dst_item->clear();
    dst_item->Resize({0});
  }
  dst_item->set_lod(src_item.lod());
}

void FetchOpHandle::RunImpl() {
  platform::RecordEvent record_event(Name(),
                                     platform::TracerEventType::Operator, 1);
  WaitInputVarGenerated(platform::CPUPlace());

  tensors_.resize(inputs_.size());
  auto &scopes = *local_exec_scopes_;

  for (size_t i = 0; i < inputs_.size(); ++i) {
    auto *var_handle = static_cast<VarHandle *>(inputs_[i]);
    auto &scope = scopes.at(var_handle->scope_idx());
    auto *var = scope->FindVar(var_handle->name());
    PADDLE_ENFORCE_NOT_NULL(
        var,
        platform::errors::NotFound(
            "Cannot find variable %s in execution scope.", var_handle->name()));

    if (var->IsType<LoDTensor>()) {
      auto &t = var->Get<framework::LoDTensor>();
      auto &item = BOOST_GET(LoDTensor, tensors_[i]);
      TransData(t, &item);
    } else {
      auto &t = var->Get<framework::LoDTensorArray>();
      LoDTensorArray tmp(t.size());
      tensors_[i] = tmp;
      auto &item = BOOST_GET(LoDTensorArray, tensors_[i]);
      for (size_t j = 0; j < t.size(); ++j) {
        TransData(t[j], &item[j]);
      }
    }
  }
  this->WaitAndMergeCPUFetchVars();
}

void FetchOpHandle::WaitInputVarGenerated(const platform::Place &place) {
  auto cpu_ctx = platform::DeviceContextPool::Instance().Get(place);
  for (auto *input : inputs_) {
    if (input->GeneratedOp()) {
      input->GeneratedOp()->RecordWaitEventOnCtx(cpu_ctx);
    }
  }
}

bool FetchOpHandle::IsMultiDeviceTransfer() { return true; }

std::string FetchOpHandle::Name() const { return "Fetch"; }

}  // namespace details
}  // namespace framework
}  // namespace paddle
