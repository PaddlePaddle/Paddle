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
#include "paddle/fluid/framework/details/fused_all_reduce_op_handle.h"
#include <zconf.h>
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

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
FusedAllReduceOpHandle::FusedAllReduceOpHandle(
    ir::Node *node, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places, const size_t num_of_all_reduce,
    const platform::NCCLContextMap *ctxs)
    : OpHandleBase(node),
      local_scopes_(local_scopes),
      places_(places),
      num_of_all_reduce_(num_of_all_reduce),
      nccl_ctxs_(ctxs) {
  if (nccl_ctxs_) {
    for (auto &p : places_) {
      this->SetDeviceContext(p, nccl_ctxs_->DevCtx(p));
    }
  }
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
  platform::RecordEvent record_event(Name(), dev_ctxes_.cbegin()->second);

  WaitInputVarGenerated();
  auto in_var_handles = DynamicCast<VarHandle>(this->Inputs());
  auto out_var_handles = DynamicCast<VarHandle>(this->Outputs());

  size_t place_num = places_.size();
  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), place_num * num_of_all_reduce_,
      "The NoDummyInputSize should be equal to the number of places.");
  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), out_var_handles.size(),
      "The NoDummyInputSize and NoDummyOutputSize should be equal.");

  std::vector<std::vector<std::pair<std::string, framework::LoDTensor>>>
      grads_tensor;
  grads_tensor.resize(place_num);
  int64_t total_num = -1;
  auto fuse_space_type = static_cast<framework::proto::VarType::Type>(0);
  for (size_t i = 0; i < local_scopes_.size(); ++i) {
    auto *s = local_scopes_[i];
    Scope &local_scope = *s->FindVar(kLocalExecScopeName)->Get<Scope *>();
    auto &vec = grads_tensor.at(i);
    // TODO(zcd): check place
    for (size_t j = 0; j < in_var_handles.size(); j += num_of_all_reduce_) {
      auto var_name = in_var_handles[j]->name();
      PADDLE_ENFORCE_EQ(var_name, out_var_handles[j]->name());
      auto &lod_tensor =
          local_scope.FindVar(var_name)->Get<framework::LoDTensor>();
      vec.emplace_back(std::make_pair(var_name, lod_tensor));
    }

    // Check the dtype of the input
    int64_t element_num = 0;
    for (size_t k = 0; k < vec.size(); ++k) {
      int64_t len = vec.at(k - 1).second.numel();
      PADDLE_ENFORCE_GT(len, 0);
      element_num += len;
      auto dtype = vec.at(k - 1).second.type();
      if (fuse_space_type == static_cast<framework::proto::VarType::Type>(0)) {
        fuse_space_type = dtype;
        PADDLE_ENFORCE_NE(dtype,
                          static_cast<framework::proto::VarType::Type>(0));
      }
      PADDLE_ENFORCE_EQ(dtype, fuse_space_type);
    }

    if (total_num == -1) {
      total_num = element_num;
    }
    PADDLE_ENFORCE_EQ(total_num, element_num);

    // Check the continuity of address space
    sort(vec.begin(), vec.end(),
         [](std::pair<std::string, framework::LoDTensor> &a,
            std::pair<std::string, framework::LoDTensor> &b) -> bool {
           return a.second.data<void>() > b.second.data<void>();
         });
    auto place = vec.at(0).second.place();
    for (size_t k = 1; k < vec.size(); ++k) {
      PADDLE_ENFORCE_EQ(place, vec.at(k).second.place());
      void *pre_address = vec.at(k - 1).second.data<void>();
      int64_t len = vec.at(k - 1).second.numel();
      void *cur_address = vec.at(k).second.data<void>();
      auto offset = len * framework::SizeOfType(fuse_space_type);
      PADDLE_ENFORCE_EQ(reinterpret_cast<void *>(
                            reinterpret_cast<uintptr_t>(pre_address) + offset),
                        cur_address);
    }
  }

  if (!FLAGS_skip_fused_all_reduce_check) {
    for (size_t j = 0; j < num_of_all_reduce_; ++j) {
      for (size_t i = 1; i < place_num; ++i) {
        PADDLE_ENFORCE_EQ(grads_tensor.at(j).at(0).first,
                          grads_tensor.at(j).at(i).first);
      }
    }
  }

  std::vector<LoDTensor *> lod_tensors;
  auto &tensor_0 = grads_tensor.at(0).at(0).second;
  auto place = tensor_0.place();
  auto dtype = tensor_0.type();
  lod_tensors.emplace_back(&(tensor_0));
  for (size_t i = 1; i < place_num; ++i) {
    auto &tensor = grads_tensor.at(i).at(0).second;
    PADDLE_ENFORCE_EQ(place, tensor.place());
    PADDLE_ENFORCE_EQ(dtype, tensor.type());
    lod_tensors.emplace_back(&tensor);
  }

  if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    PADDLE_ENFORCE(nccl_ctxs_, "nccl_ctxs should not be nullptr.");
    int nccl_dtype = platform::ToNCCLDataType(dtype);
    size_t numel = total_num;
    std::vector<std::function<void()>> all_reduce_calls;
    for (size_t i = 0; i < local_scopes_.size(); ++i) {
      auto &p = places_[i];
      void *buffer = lod_tensors.at(i)->data<void>();

      int dev_id = boost::get<platform::CUDAPlace>(p).device;
      auto &nccl_ctx = nccl_ctxs_->at(dev_id);
      auto stream = nccl_ctx.stream();
      auto comm = nccl_ctx.comm_;
      all_reduce_calls.emplace_back([=] {
        PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
            buffer, buffer, numel, static_cast<ncclDataType_t>(nccl_dtype),
            ncclSum, comm, stream));
      });
    }

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

//    if (FLAGS_sync_nccl_allreduce) {
//      for (auto &p : places_) {
//        int dev_id = boost::get<platform::CUDAPlace>(p).device;
//        auto &nccl_ctx = nccl_ctxs_->at(dev_id);
//        auto stream = nccl_ctx.stream();
//        cudaStreamSynchronize(stream);
//      }
//    }
#else
    PADDLE_THROW("Not compiled with CUDA");
#endif
  } else {
    PADDLE_THROW("Not support");
    //      // Special handle CPU only Operator's gradient. Like CRF
    //      auto &trg = *this->local_scopes_[0]
    //                       ->FindVar(kLocalExecScopeName)
    //                       ->Get<Scope *>()
    //                       ->FindVar(out_var_handles[0]->name())
    //                       ->GetMutable<framework::LoDTensor>();
    //
    //      // Reduce All Tensor to trg in CPU
    //      ReduceLoDTensor func(lod_tensors, &trg);
    //      VisitDataType(lod_tensors[0]->type(), func);
    //
    //      for (size_t i = 1; i < local_scopes_.size(); ++i) {
    //        auto &scope =
    //            *local_scopes_[i]->FindVar(kLocalExecScopeName)->Get<Scope
    //            *>();
    //        auto &p = places_[i];
    //        auto *var = scope.FindVar(out_var_handles[i]->name());
    //        auto *dev_ctx = dev_ctxes_.at(p);
    //
    //        RunAndRecordEvent(p, [&trg, var, dev_ctx, p] {
    //          auto &tensor_gpu = *var->GetMutable<framework::LoDTensor>();
    //          auto &tensor_cpu = trg;
    //          TensorCopy(tensor_cpu, p, *dev_ctx, &tensor_gpu);
    //        });
    //      }
  }
}

std::string FusedAllReduceOpHandle::Name() const { return "fused_all_reduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
