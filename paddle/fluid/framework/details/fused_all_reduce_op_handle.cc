//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/details/fused_all_reduce_op_handle.h"
#include <algorithm>
#include <utility>
#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/reduce_and_gather.h"
#include "paddle/fluid/framework/details/variable_visitor.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_bool(skip_fused_all_reduce_check, false, "");
namespace paddle {
namespace framework {
namespace details {

// Note(zcd): Addresses should be aligned, otherwise, the results may have
// diff.
static size_t Alignment(size_t size, const platform::Place &place) {
  // Allow to allocate the minimum chunk size is 4 KB.
  size_t alignment = 1 << 12;
  if (platform::is_gpu_place(place)) {
    // Allow to allocate the minimum chunk size is 256 B.
    alignment = 1 << 8;
  }
  size_t remaining = size % alignment;
  return remaining == 0 ? size : size + (alignment - remaining);
}

typedef std::vector<std::vector<std::pair<std::string, const LoDTensor *>>>
    GradientAndLoDTensor;

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
FusedAllReduceOpHandle::FusedAllReduceOpHandle(
    ir::Node *node, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places, const size_t num_of_all_reduce,
    const platform::NCCLCommunicator *ctxs)
    : NCCLOpHandleBase(node, places, ctxs),
      local_scopes_(local_scopes),
      num_of_all_reduce_(num_of_all_reduce) {
  PADDLE_ENFORCE_EQ(places_.size(), local_scopes_.size());
}
#else

FusedAllReduceOpHandle::FusedAllReduceOpHandle(
    ir::Node *node, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places, const size_t num_of_all_reduce)
    : OpHandleBase(node),
      local_scopes_(local_scopes),
      places_(places),
      num_of_all_reduce_(num_of_all_reduce) {
  PADDLE_ENFORCE_EQ(places_.size(), local_scopes_.size());
}

#endif

void FusedAllReduceOpHandle::RunImpl() {
  platform::RecordEvent record_event(Name());

  VLOG(4) << this->DebugString();

  WaitInputVarGenerated();
  // The input: grad0(dev0), grad0(dev1), grad1(dev0), grad1(dev1)...
  // The output: grad0(dev0), grad0(dev1), grad1(dev0), grad1(dev1)...
  auto in_var_handles = DynamicCast<VarHandle>(this->Inputs());
  auto out_var_handles = DynamicCast<VarHandle>(this->Outputs());

  size_t place_num = places_.size();
  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), place_num * num_of_all_reduce_,
      "The NoDummyInputSize should be equal to the number of places.");
  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), out_var_handles.size(),
      "The NoDummyInputSize and NoDummyOutputSize should be equal.");

  GradientAndLoDTensor grads_tensor;
  grads_tensor.resize(place_num);

  int64_t numel = -1;
  auto dtype = static_cast<framework::proto::VarType::Type>(0);
  for (size_t scope_idx = 0; scope_idx < local_scopes_.size(); ++scope_idx) {
    auto &g_tensor = grads_tensor.at(scope_idx);
    g_tensor.reserve(num_of_all_reduce_);

    GetGradLoDTensor(scope_idx, in_var_handles, out_var_handles, &g_tensor);

    int64_t element_num = 0;
    framework::proto::VarType::Type ele_dtype =
        static_cast<framework::proto::VarType::Type>(0);
    GetDTypeAndNumel(g_tensor, &ele_dtype, &element_num);

    if (numel == -1) {
      numel = element_num;
    }
    if (dtype == static_cast<framework::proto::VarType::Type>(0)) {
      dtype = ele_dtype;
      PADDLE_ENFORCE_NE(ele_dtype,
                        static_cast<framework::proto::VarType::Type>(0));
    }
    PADDLE_ENFORCE_EQ(ele_dtype, dtype);

    // Check whether the address space is contiguous.
    std::sort(
        g_tensor.begin(), g_tensor.end(),
        [](const std::pair<std::string, const LoDTensor *> &grad1,
           const std::pair<std::string, const LoDTensor *> &grad2) -> bool {
          return grad1.second->data<void>() < grad2.second->data<void>();
        });

    size_t size_of_dtype = framework::SizeOfType(dtype);
    for (size_t k = 1; k < g_tensor.size(); ++k) {
      const void *cur_address = g_tensor.at(k - 1).second->data<void>();
      int64_t len = g_tensor.at(k - 1).second->numel();
      auto offset = Alignment(len * size_of_dtype, places_[0]);
      void *infer_next_address = reinterpret_cast<void *>(
          reinterpret_cast<uintptr_t>(cur_address) + offset);
      const void *next_address = g_tensor.at(k).second->data<void>();

      VLOG(10) << string::Sprintf(
          "Input[%d](%s) address: 0X%02x, Input[%d](%s) address: 0X%02x, Infer "
          "input[%d] address: 0X%02x. The offset: %d",
          k - 1, g_tensor.at(k - 1).first, cur_address, g_tensor.at(k).first, k,
          next_address, k, infer_next_address, offset);
      PADDLE_ENFORCE_EQ(infer_next_address, next_address,
                        "The address is not consistent.");
    }
  }

  if (!FLAGS_skip_fused_all_reduce_check) {
    for (size_t scope_idx = 0; scope_idx < place_num; ++scope_idx) {
      for (size_t j = 1; j < num_of_all_reduce_; ++j) {
        PADDLE_ENFORCE_EQ(grads_tensor.at(0).at(j).first,
                          grads_tensor.at(scope_idx).at(j).first);
      }
    }
  }

  std::vector<const void *> lod_tensor_data;
  for (size_t scope_idx = 0; scope_idx < place_num; ++scope_idx) {
    auto data = grads_tensor.at(scope_idx).at(0).second->data<void>();
    lod_tensor_data.emplace_back(data);
  }

  if (platform::is_gpu_place(places_[0])) {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    PADDLE_ENFORCE(nccl_ctxs_, "nccl_ctxs should not be nullptr.");
    int nccl_dtype = platform::ToNCCLDataType(dtype);
    std::vector<std::function<void()>> all_reduce_calls;
    for (size_t i = 0; i < local_scopes_.size(); ++i) {
      auto &p = places_[i];
      void *buffer = const_cast<void *>(lod_tensor_data.at(i));

      all_reduce_calls.emplace_back([=] {
        NCCLAllReduce(p, buffer, buffer, numel,
                      static_cast<ncclDataType_t>(nccl_dtype), ncclSum);
      });
    }

    VLOG(10) << "fusedallreduce size:" << numel * SizeOfType(dtype);

    this->RunAndRecordEvent([&] {
      if (all_reduce_calls.size() == 1UL) {
        // Do not use NCCLGroup when manage NCCL by per thread per device
        all_reduce_calls[0]();
      } else {
        platform::NCCLGroupGuard guard;
        for (auto &call : all_reduce_calls) {
          call();
        }
      }
    });
#else
    PADDLE_THROW("Not compiled with CUDA");
#endif
  } else {
    // Special handle CPU only Operator's gradient. Like CRF
    auto grad_name = grads_tensor.at(0).at(0).first;
    auto &trg = *this->local_scopes_[0]
                     ->FindVar(kLocalExecScopeName)
                     ->Get<Scope *>()
                     ->FindVar(grad_name)
                     ->GetMutable<framework::LoDTensor>();

    // Reduce All data to trg in CPU
    ReduceBufferData func(lod_tensor_data, trg.data<void>(), numel);
    VisitDataType(trg.type(), func);

    for (size_t i = 1; i < local_scopes_.size(); ++i) {
      auto &scope =
          *local_scopes_[i]->FindVar(kLocalExecScopeName)->Get<Scope *>();
      auto &p = places_[i];
      auto *var = scope.FindVar(grad_name);
      auto *dev_ctx = dev_ctxes_.at(p);
      size_t size = numel * SizeOfType(trg.type());
      RunAndRecordEvent(p, [&trg, var, dev_ctx, p, size] {
        auto dst_ptr = var->GetMutable<framework::LoDTensor>()->data<void>();
        platform::CPUPlace cpu_place;
        memory::Copy(cpu_place, dst_ptr, cpu_place, trg.data<void>(), size);
      });
    }
  }
}

void FusedAllReduceOpHandle::GetGradLoDTensor(
    const size_t &scope_idx, const std::vector<VarHandle *> &in_var_handles,
    const std::vector<VarHandle *> &out_var_handles,
    std::vector<std::pair<std::string, const LoDTensor *>> *grad_tensor) const {
  auto *local_scope =
      local_scopes_.at(scope_idx)->FindVar(kLocalExecScopeName)->Get<Scope *>();
  size_t place_num = places_.size();

  for (size_t j = 0; j < in_var_handles.size(); j += place_num) {
    auto var_name = in_var_handles[j]->name();
    PADDLE_ENFORCE_EQ(var_name, out_var_handles[j]->name());
    auto &lod_tensor = local_scope->FindVar(var_name)->Get<LoDTensor>();
    PADDLE_ENFORCE_EQ(lod_tensor.place(), places_.at(scope_idx));
    grad_tensor->emplace_back(std::make_pair(var_name, &lod_tensor));
  }
}

void FusedAllReduceOpHandle::GetDTypeAndNumel(
    const std::vector<std::pair<std::string, const LoDTensor *>> &grad_tensor,
    proto::VarType::Type *dtype, int64_t *numel) const {
  *numel = 0;
  size_t size_of_dtype = 0;
  for (size_t i = 0; i < grad_tensor.size(); ++i) {
    // Get dtype
    auto ele_type = grad_tensor.at(i).second->type();
    if (i == 0) {
      *dtype = ele_type;
      size_of_dtype = framework::SizeOfType(ele_type);
    }
    PADDLE_ENFORCE_EQ(ele_type, *dtype);

    // Get element number
    int64_t len = grad_tensor.at(i).second->numel();
    PADDLE_ENFORCE_GT(len, 0);
    //    Alignment(len)
    *numel += Alignment(len * size_of_dtype, places_[0]) / size_of_dtype;
  }
}

std::string FusedAllReduceOpHandle::Name() const { return "fused_all_reduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
