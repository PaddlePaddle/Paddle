// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/data_balance_op_handle.h"
#include <algorithm>
#include "paddle/fluid/framework/details/container_cast.h"

namespace paddle {
namespace framework {
namespace details {

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
DataBalanceOpHandle::DataBalanceOpHandle(
    ir::Node *node, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places,
    const platform::NCCLContextMap *ctxs)
    : OpHandleBase(node), local_scopes_(local_scopes), places_(places) {
  if (ctxs) {
    for (auto &p : places_) {
      this->SetDeviceContext(p, ctxs->DevCtx(p));
    }
  }
}
#else
DataBalanceOpHandle::DataBalanceOpHandle(
    ir::Node *node, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places)
    : OpHandleBase(node), local_scopes_(local_scopes), places_(places) {}
#endif

std::string DataBalanceOpHandle::Name() const { return "data balance"; }

std::vector<std::array<int, 3>> DataBalanceOpHandle::GetBalancePlan(
    const std::vector<int> &device_sizes) {
  int device_num = device_sizes.size();
  int total_size = 0;
  int empty_num = 0;
  std::vector<std::array<int, 2>> size_device_vec;
  size_device_vec.reserve(device_num);
  for (int i = 0; i < device_num; ++i) {
    if (device_sizes[i] == 0) {
      ++empty_num;
    }
    total_size += device_sizes[i];
    size_device_vec.push_back({{device_sizes[i], i}});
  }
  std::vector<std::array<int, 3>> res;
  if (empty_num == 0) {
    // No need to do data balance.
    return res;
  }
  if (total_size < device_num) {
    // No enough data.
    PADDLE_THROW_EOF();
  }
  std::sort(size_device_vec.begin(), size_device_vec.end(),
            [](const std::array<int, 2> &a, const std::array<int, 2> &b) {
              return a[0] > b[0];
            });
  int expected_device_size = total_size / device_num;
  int src_idx = 0;
  for (int dst_idx = device_num - empty_num; dst_idx < device_num; ++dst_idx) {
    if (size_device_vec[src_idx][0] <= expected_device_size) {
      ++src_idx;
      PADDLE_ENFORCE_LT(
          src_idx, device_num - empty_num,
          "In current srategy an empty tensor should not be copy source.");
    }
    size_device_vec[src_idx][0] -= expected_device_size;
    size_device_vec[dst_idx][0] += expected_device_size;
    res.push_back({{size_device_vec[src_idx][1], size_device_vec[dst_idx][1],
                    expected_device_size}});
  }
  return res;
}

void DataBalanceOpHandle::RunImpl() {
  PADDLE_ENFORCE_GT(places_.size(), 1UL,
                    "Data balance can only be enabled when the number of "
                    "places to run larger than 1.");
  auto in_var_handles = DynamicCast<VarHandle>(this->Inputs());
  auto out_var_handles = DynamicCast<VarHandle>(this->Outputs());
  PADDLE_ENFORCE(in_var_handles.size() % places_.size() == 0);
  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), out_var_handles.size(),
      "The NoDummyInputSize and NoDummyOutputSize should be equal.");
  int data_num = in_var_handles.size() / places_.size();
  WaitInputVarGenerated();
  std::vector<std::vector<LoDTensor *>> lod_tensors(data_num);
  std::vector<int> device_sizes;
  for (int i = 0; i < static_cast<int>(in_var_handles.size()); ++i) {
    PADDLE_ENFORCE_EQ(in_var_handles[i]->name(), out_var_handles[i]->name(),
                      "The name of input and output should be equal.");
    int place_idx = i / data_num;
    int data_idx = i % data_num;
    auto *local_scope =
        local_scopes_[place_idx]->FindVar(kLocalExecScopeName)->Get<Scope *>();
    auto *tensor_var = local_scope->FindVar(in_var_handles[i]->name());
    PADDLE_ENFORCE(tensor_var->IsType<LoDTensor>());
    auto *tensor = tensor_var->GetMutable<LoDTensor>();
    lod_tensors[data_idx].push_back(tensor);
    int ins_size =
        tensor->lod().empty() ? tensor->dims()[0] : tensor->NumElements();
    if (data_idx == 0) {
      device_sizes.emplace_back(ins_size);
    } else {
      PADDLE_ENFORCE_EQ(
          ins_size, device_sizes.at(place_idx),
          "All data on the same device shall have the same batch size.");
    }
  }
  const auto &balance_plan = GetBalancePlan(device_sizes);

  for (const auto &trans : balance_plan) {
    for (int data_idx = 0; data_idx < data_num; ++data_idx) {
      LoDTensor *src_tensor = lod_tensors[data_idx][trans[0]];
      LoDTensor *dst_tensor = lod_tensors[data_idx][trans[1]];
      int trans_ins_size = trans[2];
      LoD src_lod = src_tensor->lod();
      int src_ins_size =
          src_lod.empty() ? src_tensor->dims()[0] : src_tensor->NumElements();
      int cut_point = src_ins_size - trans_ins_size;
      if (!src_lod.empty()) {
        for (auto &level : src_lod) {
          cut_point = level[cut_point];
        }
      }
      TensorCopySync(src_tensor->Slice(cut_point, src_tensor->dims()[0]),
                     dst_tensor->place(), dst_tensor);
      src_tensor->ShareDataWith(src_tensor->Slice(0, cut_point));
      if (!src_lod.empty()) {
        dst_tensor->set_lod(SliceInLevel(
            src_lod, 0, src_ins_size - trans_ins_size, src_ins_size));
        src_tensor->set_lod(
            SliceInLevel(src_lod, 0, 0, src_ins_size - trans_ins_size));
      }
    }
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
