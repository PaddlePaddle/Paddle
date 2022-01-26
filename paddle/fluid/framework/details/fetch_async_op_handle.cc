//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/fetch_async_op_handle.h"

#include <string>

#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace framework {
namespace details {

FetchAsyncOpHandle::FetchAsyncOpHandle(ir::Node *node, FetchResultType *data,
                                       size_t offset,
                                       std::vector<Scope *> *local_scopes,
                                       std::vector<Scope *> *local_exec_scopes,
                                       bool return_merged)
    : OpHandleBase(node),
      data_(data),
      offset_(offset),
      local_scopes_(local_scopes),
      local_exec_scopes_(local_exec_scopes),
      return_merged_(return_merged) {}

FetchAsyncOpHandle::~FetchAsyncOpHandle() {}

void FetchAsyncOpHandle::RecordWaitEventOnCtx(
    platform::DeviceContext *waited_ctx) {
  PADDLE_THROW(platform::errors::PermissionDenied(
      "No nodes need to wait FetchAsyncOp. Unexpceted Error."));
}

static void CheckTensorAttrs(const LoDTensor *tensor,
                             const proto::VarType::Type &type,
                             const DataLayout &layout, const DDim &dims,
                             const LoD &lod, const size_t offset) {
  if (tensor->numel() && tensor->IsInitialized()) {
    // step1: check type
    PADDLE_ENFORCE_EQ(
        type, tensor->type(),
        platform::errors::InvalidArgument(
            "The data type of fetched Tensors or the items of fetched "
            "LoDTensorArray are different from each other on different "
            "devices(%s vs %s). And the error is caused by the %zu "
            "(th) fetched variable. Please set the "
            "parameter `return_merged = False` when you "
            "call the `Executor.run()` method.",
            DataTypeToString(type), DataTypeToString(tensor->type()), offset));

    // step2: check layout
    PADDLE_ENFORCE_EQ(
        layout, tensor->layout(),
        platform::errors::InvalidArgument(
            "The layout of fetched Tensors or the items of fetched "
            "LoDTensorArray are different from each other on different "
            "devices(%s vs %s). And the error is caused by the %zu "
            "(th) fetched variable. Please set the "
            "parameter `return_merged = False` when you "
            "call the `Executor.run()` method.",
            DataLayoutToString(layout), DataLayoutToString(tensor->layout()),
            offset));
  }

  // step3: check dims
  auto tensor_dims = tensor->dims();
  PADDLE_ENFORCE_EQ(dims.size(), tensor_dims.size(),
                    platform::errors::InvalidArgument(
                        "The dimension sizes of fetched Tensors or "
                        "the items of fetched LoDTensorArray are "
                        "different from each other on different "
                        "devices(%s vs %s). And the error is caused by the %zu "
                        "(th) fetched variable. Please set the "
                        "parameter `return_merged = False` when you "
                        "call the `Executor.run()` method.",
                        dims, tensor_dims, offset));
  for (int j = 1; j < dims.size(); j++) {
    PADDLE_ENFORCE_EQ(dims[j], tensor_dims[j],
                      platform::errors::InvalidArgument(
                          "The dimensions of fetched Tensors or "
                          "the items of fetched LoDTensorArray are "
                          "different from each other on different "
                          "devices(%s vs %s). And the error is caused by the "
                          "%zu (th) fetched variable. Please set the "
                          "parameter `return_merged = False` when "
                          "you call the `Executor.run()` method.",
                          dims, tensor_dims, offset));
  }

  // step4: check lod
  PADDLE_ENFORCE_EQ(
      lod.size(), tensor->lod().size(),
      platform::errors::InvalidArgument(
          "The LoD information of fetched Tensors or the items of fetched "
          "LoDTensorArray are different from each other on different "
          "devices(%s vs %s). And the error is caused by the %zu "
          "(th) fetched variable. Please set the "
          "parameter `return_merged = False` when you "
          "call the `Executor.run()` method.",
          lod, tensor->lod(), offset));
}

static void TransData(const framework::Tensor *src_item,
                      framework::Tensor *dst_item,
                      const platform::DeviceContext &ctx) {
  if (src_item->IsInitialized() && src_item->numel() > 0) {
    if (platform::is_gpu_place(src_item->place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      TensorCopy(*src_item, platform::CUDAPinnedPlace(), ctx, dst_item);
#endif
    } else {
      TensorCopy(*src_item, platform::CPUPlace(), dst_item);
    }
  }
}

void FetchAsyncOpHandle::FetchMergedLodTensor(
    const std::vector<const LoDTensor *> &src_lodtensors,
    LoDTensor *dst_lodtensor) {
  // calc dst type,layout,dim,lod and calc check dim
  proto::VarType::Type new_type = proto::VarType::FP32;
  framework::DataLayout new_layout;
  framework::DDim new_dim;
  LoD new_lod = src_lodtensors[0]->lod();

  framework::DDim check_dim;

  for (auto *t : src_lodtensors) {
    if (t->numel() && t->IsInitialized()) {
      check_dim = t->dims();
      new_type = t->type();
      new_layout = t->layout();
      break;
    }
  }

  bool find_first_dims = false;
  for (auto *t : src_lodtensors) {
    if (t->numel() && t->IsInitialized()) {
      if (!find_first_dims) {
        new_dim = t->dims();
        find_first_dims = true;
      } else {
        new_dim[0] += t->dims()[0];
      }
    }
  }

  // check src type,layout,dim,lod consistence
  for (size_t i = 1; i < src_lodtensors.size(); ++i) {
    CheckTensorAttrs(src_lodtensors[i], new_type, new_layout, check_dim,
                     new_lod, offset_);
  }

  // set dst tensor
  dst_lodtensor->Resize(new_dim);
  dst_lodtensor->set_layout(src_lodtensors[0]->layout());
  dst_lodtensor->set_lod(src_lodtensors[0]->lod());
  if (platform::is_gpu_place(src_lodtensors[0]->place())) {
    dst_lodtensor->mutable_data(platform::CUDAPinnedPlace(),
                                src_lodtensors[0]->type());
  } else {
    dst_lodtensor->mutable_data(platform::CPUPlace(),
                                src_lodtensors[0]->type());
  }

  // slice and memcpy
  int begin = 0;
  for (auto *src : src_lodtensors) {
    int end = begin + src->dims()[0];
    if (end == begin) {
      continue;
    }
    auto dst = dst_lodtensor->Slice(begin, end);
    TransData(src, &dst, *dev_ctxes_[src->place()]);
    begin = end;
  }
}

void FetchAsyncOpHandle::RunImpl() {
  platform::RecordEvent record_event(Name());
  WaitInputVarGenerated(true);

  // get src vars
  auto &scopes = *local_exec_scopes_;
  std::vector<Variable *> src_vars;
  src_vars.reserve(inputs_.size());
  for (size_t i = 0; i < inputs_.size(); ++i) {
    auto *var_handle = static_cast<VarHandle *>(inputs_[i]);
    auto &scope = scopes.at(var_handle->scope_idx());
    auto *var = scope->FindVar(var_handle->name());
    PADDLE_ENFORCE_NOT_NULL(
        var,
        platform::errors::NotFound(
            "Cannot find variable %s in execution scope.", var_handle->name()));
    src_vars.emplace_back(var);
  }

  if (return_merged_) {
    auto &val = BOOST_GET(FetchList, *data_);
    if (src_vars[0]->IsType<LoDTensor>()) {
      // to lodtensor type
      std::vector<const LoDTensor *> src_lodtensors;
      src_lodtensors.reserve(src_vars.size());
      for (size_t i = 0; i < src_vars.size(); ++i) {
        src_lodtensors.emplace_back(&src_vars[i]->Get<framework::LoDTensor>());
      }

      LoDTensor dst_lodtensor;
      FetchMergedLodTensor(src_lodtensors, &dst_lodtensor);
      val.at(offset_) = std::move(dst_lodtensor);
    } else {
      // to lodtensorarray type
      std::vector<const LoDTensorArray *> src_lodtensor_arrays;
      src_lodtensor_arrays.reserve(src_vars.size());
      for (size_t i = 0; i < src_vars.size(); ++i) {
        src_lodtensor_arrays.emplace_back(
            &src_vars[i]->Get<framework::LoDTensorArray>());
      }

      LoDTensorArray dst_lodtensor_array;
      dst_lodtensor_array.resize(src_lodtensor_arrays[0]->size());

      for (size_t i = 0; i < dst_lodtensor_array.size(); ++i) {
        std::vector<const LoDTensor *> src_lodtensors;
        src_lodtensors.reserve(src_lodtensor_arrays.size());
        for (size_t j = 0; j < src_lodtensor_arrays.size(); ++j) {
          src_lodtensors.emplace_back(&(*src_lodtensor_arrays[j])[i]);
        }
        FetchMergedLodTensor(src_lodtensors, &dst_lodtensor_array[i]);
      }
      val.at(offset_) = std::move(dst_lodtensor_array);
    }
  } else {
    auto &val = BOOST_GET(FetchUnmergedList, *data_);
    auto &dst_tensors = val.at(offset_);
    dst_tensors.reserve(src_vars.size());

    for (size_t i = 0; i < src_vars.size(); ++i) {
      if (src_vars[i]->IsType<LoDTensor>()) {
        auto &t = src_vars[i]->Get<framework::LoDTensor>();
        LoDTensor item;
        TransData(&t, &item, *dev_ctxes_[t.place()]);
        dst_tensors.emplace_back(std::move(item));
      } else {
        auto &t = src_vars[i]->Get<framework::LoDTensorArray>();
        LoDTensorArray item;
        item.resize(t.size());
        for (size_t j = 0; j < t.size(); ++j) {
          TransData(&t[j], &item[j], *dev_ctxes_[t[j].place()]);
        }
        dst_tensors.emplace_back(std::move(item));
      }
    }
  }
}

bool FetchAsyncOpHandle::IsMultiDeviceTransfer() { return true; }

std::string FetchAsyncOpHandle::Name() const { return "FetchAsync"; }

}  // namespace details
}  // namespace framework
}  // namespace paddle
