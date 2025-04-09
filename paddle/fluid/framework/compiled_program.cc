/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/compiled_program.h"

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimization_var_info.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/reference_count_pass_helper.h"
#include "paddle/phi/core/operators/reader/dense_tensor_blocking_queue.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/core/platform/cuda_device_guard.h"
#endif
#include "paddle/common/flags.h"

COMMON_DECLARE_double(eager_delete_tensor_gb);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
COMMON_DECLARE_bool(sync_nccl_allreduce);
#endif

namespace paddle {
namespace framework {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
std::once_flag p2p_init_flag;
#endif

static std::unordered_set<std::string> ReaderOpSet() {
  return {"create_py_reader"};
}

class CompiledProgramPrivate {
 public:
  CompiledProgramPrivate(const std::vector<phi::Place> &places,
                         Scope *global_scope)
      : places_(places), global_scope_(global_scope) {}

  ~CompiledProgramPrivate() {
    if (own_local_scope_) {
      for (size_t i = 1; i < local_scopes_.size(); ++i) {
        // Skip the first scope, since it is the global scope.
        Scope *local_scope = local_scopes_[i];
        if (global_scope_->HasKid(local_scope)) {
          global_scope_->DeleteScope(local_scope);
        }
      }
    }
  }

  bool IsUseCUDA(DeviceType use_device);

  ir::Graph *ApplyMemoryOptimizePass(ir::Graph *graph);

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  void InitNCCLCtxs(framework::Scope *scope, const BuildStrategy &bst) {
    VLOG(1) << "nccl comm num:" << bst.nccl_comm_num_ << ", nranks:" << nranks_
            << ", num_trainers:" << bst.num_trainers_
            << ", trainer_id:" << bst.trainer_id_;

    if (bst.use_hierarchical_allreduce_) {
      VLOG(1) << ", use_hierarchical_allreduce:"
              << bst.use_hierarchical_allreduce_ << ", inter_trainers_num:"
              << bst.hierarchical_allreduce_inter_nranks_
              << ", exter_trainers_num:"
              << bst.hierarchical_allreduce_exter_nranks_;
    }

    std::vector<ncclUniqueId *> flat_nccl_ids;
    if (nranks_ == 1) {
      // FIXME(gongwb): need not to create ncclid when nranks==1
      nccl_ctxs_->InitFlatCtxs(
          places_, flat_nccl_ids, bst.num_trainers_, bst.trainer_id_);
      return;
    }

    if (bst.enable_parallel_graph_) {
      VLOG(1) << "use only one ncclid in pg model";

      ncclUniqueId *nccl_id = nullptr;

      std::string var_name = platform::GetFlatNCCLVarName(0);
      auto nccl_id_var = scope->FindVar(var_name);
      if (nccl_id_var) {
        nccl_id = nccl_id_var->GetMutable<ncclUniqueId>();
        VLOG(10) << "find nccl_id_var:" << var_name << ", nccl_id:" << nccl_id;
      } else {
        nccl_id = new ncclUniqueId();
        PADDLE_ENFORCE_EQ(
            phi::dynload::ncclGetUniqueId(nccl_id),
            ncclSuccess,
            common::errors::PreconditionNotMet(
                "PaddlePaddle failed to get NCCL unique ID. It may due to your "
                "system settings or NCCL library error, please debug on NCCL"));
        VLOG(10) << "can't find nccl_id_var:" << var_name
                 << ", nccl_id:" << nccl_id;
      }

      flat_nccl_ids.push_back(nccl_id);

      nccl_ctxs_->InitFlatCtxs(
          places_, flat_nccl_ids, bst.num_trainers_, bst.trainer_id_);
      VLOG(1) << "init bst nccl context complete!";
      return;
    }

    // num_trainers ==1 && places > 1
    if (bst.num_trainers_ == 1) {
      nccl_ctxs_->InitFlatCtxs(
          places_, flat_nccl_ids, bst.num_trainers_, bst.trainer_id_);
      return;
    }

    for (int i = 0; i < static_cast<int>(bst.nccl_comm_num_); i++) {
      std::string var_name = platform::GetFlatNCCLVarName(i);
      auto nccl_id_var = scope->FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(
          nccl_id_var,
          common::errors::NotFound("Can't find nccl_id_var '%s'.", var_name));
      auto nccl_id = nccl_id_var->GetMutable<ncclUniqueId>();
      flat_nccl_ids.push_back(nccl_id);
    }

    nccl_ctxs_->InitFlatCtxs(
        places_, flat_nccl_ids, bst.num_trainers_, bst.trainer_id_);

    if (bst.use_hierarchical_allreduce_) {
      std::vector<ncclUniqueId *> inter_nccl_ids;
      for (int i = 0; i < static_cast<int>(bst.nccl_comm_num_); i++) {
        std::string var_name = platform::GetHierarchicalInterNCCLVarName(i);
        auto nccl_id_var = scope->FindVar(var_name);
        PADDLE_ENFORCE_NOT_NULL(
            nccl_id_var,
            common::errors::NotFound("Can't find nccl_id_var '%s'.", var_name));
        auto inter_nccl_id = nccl_id_var->GetMutable<ncclUniqueId>();
        inter_nccl_ids.push_back(inter_nccl_id);
      }

      std::vector<ncclUniqueId *> exter_nccl_ids;
      for (int i = 0; i < static_cast<int>(bst.nccl_comm_num_); i++) {
        std::string var_name = platform::GetHierarchicalExterNCCLVarName(i);
        auto nccl_id_var = scope->FindVar(var_name);
        PADDLE_ENFORCE_NOT_NULL(
            nccl_id_var,
            common::errors::NotFound("Can't find nccl_id_var '%s'.", var_name));
        auto nccl_id = nccl_id_var->GetMutable<ncclUniqueId>();
        exter_nccl_ids.push_back(nccl_id);
      }

      nccl_ctxs_->InitHierarchicalCtxs(
          places_,
          inter_nccl_ids,
          exter_nccl_ids,
          bst.num_trainers_,
          bst.trainer_id_,
          bst.hierarchical_allreduce_inter_nranks_,
          bst.hierarchical_allreduce_exter_nranks_);
    }
  }

  void InitOrGetNCCLCommunicator(framework::Scope *scope, BuildStrategy *bst) {
    const std::string var_name = "NCCLCommunicator";
    auto var = scope->FindVar(var_name);
    if (var != nullptr) {
      PADDLE_ENFORCE_EQ(var->IsInitialized(),
                        true,
                        common::errors::PreconditionNotMet(
                            "if %s exists, it must be initialized", var_name));
      VLOG(1) << "find " << var_name
              << " in scope, so use it and does not recreate!";
      nccl_ctxs_ = var->GetMutable<platform::NCCLCommunicator>();
      return;
    }

    if (bst->use_hierarchical_allreduce_) {
      PADDLE_ENFORCE_GT(
          bst->num_trainers_,
          1,
          common::errors::PreconditionNotMet(
              "The num_trainers should be greater than 1, but received %llu.",
              bst->num_trainers_));
      PADDLE_ENFORCE_GT(
          bst->hierarchical_allreduce_inter_nranks_,
          1,
          common::errors::PreconditionNotMet(
              "The inter_nranks should be greater than 1, but received %d.",
              bst->hierarchical_allreduce_inter_nranks_));
      PADDLE_ENFORCE_EQ(
          bst->num_trainers_ % bst->hierarchical_allreduce_inter_nranks_,
          0,
          common::errors::PreconditionNotMet(
              "num_trainers:%llu mod inter_nranks:%d != 0",
              bst->num_trainers_,
              bst->hierarchical_allreduce_inter_nranks_));

      bst->hierarchical_allreduce_exter_nranks_ =
          bst->num_trainers_ / bst->hierarchical_allreduce_inter_nranks_;
    }

    VLOG(1) << "not find " << var_name << " in scope, so recreate it!";
    nccl_ctxs_ = scope->Var(var_name)->GetMutable<platform::NCCLCommunicator>();
    InitNCCLCtxs(scope, *bst);
  }
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
  void InitBKCLCtxs(framework::Scope *scope, const BuildStrategy &bst) {
    VLOG(1) << "bkcl comm num:" << bst.bkcl_comm_num_ << ", nranks:" << nranks_
            << ", num_trainers:" << bst.num_trainers_
            << ", trainer_id:" << bst.trainer_id_;

    PADDLE_ENFORCE_EQ(bst.use_hierarchical_allreduce_,
                      false,
                      common::errors::Unimplemented(
                          "xpu doesn't support use_hierarchical_allreduce"));

    std::vector<BKCLUniqueId *> flat_bkcl_ids;
    if (nranks_ == 1) {
      // FIXME(gongwb): need not to create bkclid when nranks==1
      bkcl_ctxs_->InitFlatCtxs(
          places_, flat_bkcl_ids, bst.num_trainers_, bst.trainer_id_);
      return;
    }

    if (bst.enable_parallel_graph_) {
      VLOG(1) << "use only one bkclid in pg model";

      BKCLUniqueId *bkcl_id = nullptr;

      std::string var_name = platform::GetFlatBKCLVarName(0);
      auto bkcl_id_var = scope->FindVar(var_name);
      std::unique_ptr<BKCLUniqueId> id(new BKCLUniqueId());
      if (bkcl_id_var) {
        bkcl_id = bkcl_id_var->GetMutable<BKCLUniqueId>();
      } else {
        PADDLE_ENFORCE_EQ(
            bkcl_get_unique_id(id.get()),
            BKCL_SUCCESS,
            common::errors::Unavailable("bkcl get unique id failed"));
        bkcl_id = id.get();
      }

      flat_bkcl_ids.push_back(bkcl_id);

      bkcl_ctxs_->InitFlatCtxs(
          places_, flat_bkcl_ids, bst.num_trainers_, bst.trainer_id_);
      VLOG(1) << "init bst bkcl context complete!";
      return;
    }

    // num_trainers ==1 && places > 1
    if (bst.num_trainers_ == 1) {
      bkcl_ctxs_->InitFlatCtxs(
          places_, flat_bkcl_ids, bst.num_trainers_, bst.trainer_id_);
      return;
    }

    for (int i = 0; i < static_cast<int>(bst.bkcl_comm_num_); i++) {
      std::string var_name = platform::GetFlatBKCLVarName(i);
      auto bkcl_id_var = scope->FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(
          bkcl_id_var,
          common::errors::NotFound("can't find %s bkcl_id_var", var_name));
      auto bkcl_id = bkcl_id_var->GetMutable<BKCLUniqueId>();
      flat_bkcl_ids.push_back(bkcl_id);
    }

    bkcl_ctxs_->InitFlatCtxs(
        places_, flat_bkcl_ids, bst.num_trainers_, bst.trainer_id_);
  }

  void InitOrGetBKCLCommunicator(framework::Scope *scope,
                                 const BuildStrategy &bst) {
    const std::string var_name = "BKCLCommunicator";
    auto var = scope->FindVar(var_name);
    if (var != nullptr) {
      PADDLE_ENFORCE_EQ(var->IsInitialized(),
                        true,
                        common::errors::PreconditionNotMet(
                            "if %s exists, it must be initialized", var_name));
      VLOG(1) << "find " << var_name
              << " in scope, so use it and does not recreate!";
      bkcl_ctxs_ = var->GetMutable<platform::BKCLCommunicator>();
      return;
    }

    VLOG(1) << "not find " << var_name << " in scope, so recreate it!";
    bkcl_ctxs_ = scope->Var(var_name)->GetMutable<platform::BKCLCommunicator>();
    InitBKCLCtxs(scope, bst);
  }
#endif

  BuildStrategy build_strategy_;
  std::vector<phi::Place> places_;
  std::vector<Scope *> local_scopes_;
  Scope *global_scope_;  // not owned

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  platform::NCCLCommunicator *nccl_ctxs_{nullptr};
#elif defined(PADDLE_WITH_XPU_BKCL)
  platform::BKCLCommunicator *bkcl_ctxs_{nullptr};
#endif
  bool own_local_scope_;
  DeviceType use_device_ = p::kCUDA;
  bool use_all_reduce_;
  size_t nranks_;

  ir::MemOptVarInfoMapList mem_opt_var_infos_;
  ir::GarbageCollectorMap gcs_;
};

bool CompiledProgramPrivate::IsUseCUDA(DeviceType use_device) {
  return use_device == p::kCUDA;
}

ir::Graph *CompiledProgramPrivate::ApplyMemoryOptimizePass(ir::Graph *graph) {
  std::vector<ir::LastLiveOpsOfVars> last_live_ops_of_vars;
  size_t max_memory_size = static_cast<size_t>(GetEagerDeletionThreshold());

  for (size_t i = 0; i < places_.size(); ++i) {
    auto &place = places_[i];
    if (gcs_.count(place) > 0) {
      continue;
    }
    std::unique_ptr<GarbageCollector> gc;
    if (phi::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      if (IsFastEagerDeletionModeEnabled()) {
        gc = std::make_unique<UnsafeFastGPUGarbageCollector>(place,
                                                             max_memory_size);
      } else {
        gc = std::make_unique<StreamGarbageCollector>(place, max_memory_size);
      }
      VLOG(10) << "Created " << i << "-th GarbageCollector at " << place;
#else
      PADDLE_THROW(common::errors::PermissionDenied(
          "Paddle can't use CUDA device since it's not compiled with CUDA,"
          "Please recompile or reinstall Paddle with GPU support."));
#endif
    } else if (phi::is_xpu_place(place)) {
#if defined(PADDLE_WITH_XPU)
      gc = std::make_unique<XPUGarbageCollector>(place, max_memory_size);
      VLOG(10) << "Created " << i << "-th GarbageCollector at " << place;
#else
      PADDLE_THROW(common::errors::PermissionDenied(
          "Paddle can't use XPU device since it's not compiled with XPU,"
          "Please recompile or reinstall Paddle with XPU support."));
#endif
    } else if (phi::is_ipu_place(place)) {
#if defined(PADDLE_WITH_IPU)
      gc = std::make_unique<IPUGarbageCollector>(place, max_memory_size);
      VLOG(10) << "Created " << i << "-th GarbageCollector at " << place;
#else
      PADDLE_THROW(common::errors::PermissionDenied(
          "Paddle can't use IPU device since it's not compiled with IPU,"
          "Please recompile or reinstall Paddle with IPU support."));
#endif
    } else if (phi::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      if (IsFastEagerDeletionModeEnabled()) {
        gc = std::make_unique<CustomDeviceUnsafeFastGarbageCollector>(
            place, max_memory_size);
      } else {
        gc = std::make_unique<CustomStreamGarbageCollector>(place,
                                                            max_memory_size);
      }
      VLOG(10) << "Created " << i << "-th GarbageCollector at " << place;
#else
      PADDLE_THROW(common::errors::PermissionDenied(
          "Paddle can't use custom device since it's not compiled with "
          "CustomDevice,"
          "Please recompile or reinstall Paddle with CustomDevice support."));
#endif
    } else if (phi::is_cpu_place(place)) {
      gc = std::make_unique<CPUGarbageCollector>(place, max_memory_size);
      VLOG(10) << "Created GarbageCollector at " << place;
    } else {
      PADDLE_THROW(common::errors::PreconditionNotMet(
          "Unsupported place for garbage collection"));
    }
    gcs_.emplace(place, std::move(gc));
  }

  if (!gcs_.empty()) {
    auto eager_deletion_pass =
        ir::PassRegistry::Instance().Get("eager_deletion_pass");
    eager_deletion_pass->SetNotOwned(ir::kMemOptVarInfoMapList,
                                     &mem_opt_var_infos_);
    eager_deletion_pass->SetNotOwned(ir::kGarbageCollector, &gcs_);
    eager_deletion_pass->SetNotOwned(ir::kLastLiveOpsOfVars,
                                     &last_live_ops_of_vars);
    eager_deletion_pass->SetNotOwned(ir::kAllPlaces, &places_);
    graph = eager_deletion_pass->Apply(graph);
    VLOG(10) << "EagerDeletionPass Applied";
    VLOG(1) << "Garbage collection strategy is enabled, when "
            << "FLAGS_eager_delete_tensor_gb = "
            << FLAGS_eager_delete_tensor_gb;
  }
  return graph;
}

std::vector<Scope *> &CompiledProgram::GetLocalScopes() {
  return member_->local_scopes_;
}

void InitP2P(const std::vector<phi::Place> &places) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  std::call_once(p2p_init_flag, [&]() {
    int count = places.size();
    if (count <= 1) return;

    std::vector<int> devices;
    for (int i = 0; i < count; i++) {
      if (!phi::is_gpu_place(places[i])) return;

      phi::GPUPlace device = places[i];
      devices.push_back(device.GetDeviceId());
    }

    for (int i = 0; i < count; ++i) {
      for (int j = 0; j < count; ++j) {
        if (devices[i] == devices[j]) continue;
        int can_access = -1;
#ifdef PADDLE_WITH_HIP
        hipError_t ret =
            hipDeviceCanAccessPeer(&can_access, devices[i], devices[j]);
        if (ret != hipSuccess || can_access != 1) {
#else
        cudaError_t ret =
            cudaDeviceCanAccessPeer(&can_access, devices[i], devices[j]);
        if (ret != cudaSuccess || can_access != 1) {
#endif
          LOG(WARNING) << "Cannot enable P2P access from " << devices[i]
                       << " to " << devices[j];
        } else {
          platform::CUDADeviceGuard guard(devices[i]);
#ifdef PADDLE_WITH_HIP
          hipDeviceEnablePeerAccess(devices[j], 0);
#else
          cudaDeviceEnablePeerAccess(devices[j], 0);
#endif
        }
      }
    }
    VLOG(1) << "init p2p";
  });
#endif
}

CompiledProgram::CompiledProgram(const std::vector<phi::Place> &places,
                                 const std::vector<std::string> &bcast_vars,
                                 const std::string &loss_var_name,
                                 Scope *scope,
                                 const std::vector<Scope *> &local_scopes,
                                 const BuildStrategy &build_strategy,
                                 ir::Graph *graph)
    : member_(new CompiledProgramPrivate(places, scope)) {
  PADDLE_ENFORCE_EQ(
      !places.empty(),
      true,
      common::errors::Unavailable("NPU is not supported in CompiledProgram."));
  InitP2P(places);
  InitReaderQueueDeviceCount(
      graph, *(member_->global_scope_), member_->places_.size());
  // Initialize necessary info of member_ with strategy.
  InitProgramPrivateMemberInfo(build_strategy, places.size());

  // Step 1. Create local scopes and Clone graph into multi device
  CreateLocalScopes(scope, local_scopes, /*create_new*/ true);
  std::vector<ir::Graph *> graphs = CloneGraphToMultiDevices(graph);
  PrepareNCCLCommunicator(scope);

  // broadcast parameters from the 0th device to others:
  auto need_broadcast = [&]() -> bool {
    if (member_->build_strategy_.num_trainers_ > 1) {  // NOLINT
      // 1. num_tariners would be grater than 1 for nccl distributed training.
      return true;
    } else if (member_->local_scopes_.size() != 1 && local_scopes.empty()) {
      // 2. Only one trainer process, but CompiledProgram hold multiple
      // devices.
      return true;
    }
    return false;
  };
  if (need_broadcast()) {
    BCastParamsToDevices(bcast_vars, member_->build_strategy_.trainer_id_);
  }

  // Step 2. Convert main_program to SSA form and dependency graph. Also, insert
  // ncclOp
  std::vector<ir::Graph *> async_graphs =
      CompileGraphWithBuildStrategy(graph, &graphs, loss_var_name);
  graph = member_->ApplyMemoryOptimizePass(graph);
}

void CompiledProgram::BCastParamsToDevices(const std::vector<std::string> &vars,
                                           int trainer_id) const {
  VLOG(3) << "BCastParamsToDevices";
  // the initializing bcast, all vars would be bcast from device(0).
  for (auto &var : vars) {
    framework::Variable *main_var = member_->local_scopes_[0]->FindVar(var);
    if (main_var == nullptr || !main_var->IsType<phi::DenseTensor>()) {
      continue;
    }

    auto &main_tensor = main_var->Get<phi::DenseTensor>();
    if (!main_tensor.IsInitialized()) {
      VLOG(3) << "one in var not inited, return!";
      continue;
    }
    auto &dims = main_tensor.dims();
    if (phi::is_gpu_place(main_tensor.place())) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      std::vector<void *> buffers;
      buffers.reserve(member_->places_.size());
      size_t numel = main_tensor.numel();
      auto dtype = framework::TransToProtoVarType(main_tensor.dtype());
      ncclDataType_t data_type = phi::ToNCCLDataType(main_tensor.dtype());
      for (size_t i = 0; i < member_->places_.size(); ++i) {
        auto place = member_->places_[i];
        void *buffer;

        if (i == 0 && trainer_id == 0) {
          buffer = const_cast<void *>(main_tensor.data());
        } else {
          auto local_scope = member_->local_scopes_[i];
          auto *t = local_scope->Var(var)->GetMutable<phi::DenseTensor>();
          t->Resize(dims);
          buffer = t->mutable_data(place, main_tensor.dtype());
        }
        buffers.push_back(buffer);
      }

      PADDLE_ENFORCE_EQ(member_->places_.size(),
                        buffers.size(),
                        common::errors::PreconditionNotMet(
                            "variables' buffer size to bcast is %d, which is "
                            "NOT equal to places size %d",
                            buffers.size(),
                            member_->places_.size()));
      if (member_->nccl_ctxs_ != nullptr) {
        auto *nccl_ctxs = member_->nccl_ctxs_->DefaultFlatCtx();
        platform::NCCLGroupGuard guard;
        for (size_t i = 0; i < member_->places_.size(); ++i) {
          auto &nccl_ctx = nccl_ctxs->at(member_->places_[i]);
          phi::dynload::ncclBcast(buffers[i],
                                  numel,
                                  data_type,
                                  0,
                                  nccl_ctx.comm_,
                                  nccl_ctx.stream());
        }
        nccl_ctxs->WaitAll();
      } else {
        auto src_place = member_->places_[0];
        auto src_dev_ctx = static_cast<phi::GPUContext *>(
            phi::DeviceContextPool::Instance().Get(src_place));
        auto sizeof_dtype = framework::SizeOfType(dtype) * numel;
        for (size_t i = 1; i < member_->places_.size(); ++i) {
          auto dst_place = member_->places_[i];
          auto dst_dev_ctx = static_cast<phi::GPUContext *>(
              phi::DeviceContextPool::Instance().Get(dst_place));
          src_dev_ctx->Wait();
          dst_dev_ctx->Wait();
          memory::Copy(dst_place,
                       buffers[i],
                       src_place,
                       buffers[0],
                       sizeof_dtype,
                       src_dev_ctx->stream());
          src_dev_ctx->Wait();
          dst_dev_ctx->Wait();
        }
      }
#endif
    } else if (phi::is_xpu_place(main_tensor.place())) {
#if defined(PADDLE_WITH_XPU_BKCL)
      std::vector<void *> buffers;
      buffers.reserve(member_->places_.size());
      size_t numel = main_tensor.numel();
      BKCLDataType data_type = phi::ToBKCLDataType(main_tensor.dtype());
      for (size_t i = 0; i < member_->places_.size(); ++i) {
        auto place = member_->places_[i];
        void *buffer;

        if (i == 0 && trainer_id == 0) {
          buffer = const_cast<void *>(main_tensor.data());
        } else {
          auto local_scope = member_->local_scopes_[i];
          auto *t = local_scope->Var(var)->GetMutable<phi::DenseTensor>();
          t->Resize(dims);
          buffer = t->mutable_data(place, main_tensor.dtype());
        }
        buffers.push_back(buffer);
      }

      PADDLE_ENFORCE_EQ(member_->places_.size(),
                        buffers.size(),
                        common::errors::PreconditionNotMet(
                            "variables' buffer size to bcast is %d, which is "
                            "NOT equal to places size %d",
                            buffers.size(),
                            member_->places_.size()));
      {
        auto *bkcl_ctxs = member_->bkcl_ctxs_->DefaultFlatCtx();
        platform::BKCLGroupGuard guard;
        for (size_t i = 0; i < member_->places_.size(); ++i) {
          auto &bkcl_ctx = bkcl_ctxs->at(member_->places_[i]);
          PADDLE_ENFORCE_EQ(
              bkcl_broadcast(bkcl_ctx.comm(),
                             buffers[i],
                             buffers[i],
                             numel,
                             data_type,
                             0,
                             NULL),
              BKCL_SUCCESS,
              common::errors::Unavailable("bkcl_broadcast failed"));
        }
        bkcl_ctxs->WaitAll();
      }
#else
      PADDLE_THROW(
          common::errors::PreconditionNotMet("Not compiled with BKCL."));
#endif
    } else {
      phi::CPUPlace cpu;
      for (size_t i = 1; i < member_->places_.size(); ++i) {
        auto local_scope = member_->local_scopes_[i];
        auto *t = local_scope->Var(var)->GetMutable<phi::DenseTensor>();

        auto copy_memory = [&] {
          t->Resize(dims);
          t->mutable_data(cpu, main_tensor.dtype());
          paddle::framework::TensorCopy(main_tensor, cpu, t);
        };

        auto share_memory = [&] { t->ShareDataWith(main_tensor); };

        // FIXME(zcd): LR_DECAY_COUNTER should not be shared. This is a hot fix.
        if (member_->use_all_reduce_ ||
            member_->IsUseCUDA(member_->use_device_) ||
            var == "@LR_DECAY_COUNTER@") {
          copy_memory();
        } else {
          share_memory();
        }
      }
    }
  }
}

CompiledProgram::~CompiledProgram() {
  for (auto &p : member_->places_) {
    phi::DeviceContextPool::Instance().Get(p)->Wait();
  }
  delete member_;
}

void CompiledProgram::InitProgramPrivateMemberInfo(
    const BuildStrategy &build_strategy, size_t device_count) {
  member_->build_strategy_ = build_strategy;
  member_->use_all_reduce_ = member_->build_strategy_.reduce_ ==
                             BuildStrategy::ReduceStrategy::kAllReduce;
  member_->nranks_ = build_strategy.num_trainers_ * device_count;
  if (!member_->use_all_reduce_ && member_->nranks_ == 1) {
    LOG(INFO) << "If you set build_strategy.reduce with 'Reduce',"
                 "the number of places should be greater than 1.";
    member_->build_strategy_.reduce_ =
        BuildStrategy::ReduceStrategy::kAllReduce;
    member_->use_all_reduce_ = true;
  }
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && defined(_WIN32)
  if (member_->IsUseCUDA(member_->use_device_)) {
    PADDLE_ENFORCE_EQ(
        device_count,
        1,
        common::errors::Unavailable("Windows can support Single GPU only."));
  }
#endif

#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && \
    (!defined(PADDLE_WITH_NCCL) && !defined(PADDLE_WITH_RCCL))
  if (member_->IsUseCUDA(member_->use_device_)) {
    PADDLE_ENFORCE_EQ(
        device_count,
        1,
        common::errors::PermissionDenied(
            "Your machine has multiple cards, "
            "but the WITH_NCCL option is not turned on during compilation, "
            "and you cannot use multi-card training or prediction. "
            "Please recompile and turn on the WITH_NCCL option."));
  }
#endif

  std::string device_name;
  if (member_->use_device_ == p::kCPU) {
    device_name = "CPU";
  } else if (member_->use_device_ == p::kCUDA) {
    device_name = "CUDA";
  } else if (member_->use_device_ == p::kXPU) {
    device_name = "XPU";
  } else {
    PADDLE_THROW(
        common::errors::Unavailable("Only CPU/CUDA/XPU is supported. "
                                    "please use CPU/CUDA/XPU backend."));
  }
}

void CompiledProgram::InitReaderQueueDeviceCount(ir::Graph *graph,
                                                 const Scope &scope,
                                                 size_t dev_cnt) {
  using QueueHolder =
      operators::reader::OrderedMultiDeviceDenseTensorBlockingQueueHolder;

  auto reader_ops = ReaderOpSet();
  for (auto &node : graph->Nodes()) {
    if (node->IsOp() && node->Op() &&
        reader_ops.count(node->Op()->Type()) != 0) {
      auto queue_name = node->Op()->Input("blocking_queue")[0];
      auto var = scope.FindVar(queue_name);
      if (var && var->IsType<QueueHolder>()) {
        VLOG(10) << "Set device count of " << queue_name << " to be "
                 << dev_cnt;
        var->GetMutable<QueueHolder>()->GetQueue()->SetDeviceCount(dev_cnt);
      }
    }
  }
}

void CompiledProgram::CreateLocalScopes(
    Scope *global_scope,
    const std::vector<Scope *> &local_scopes,
    bool create_new) {
  if (local_scopes.empty()) {
    member_->own_local_scope_ = true;
    member_->local_scopes_.emplace_back(global_scope);
    for (size_t i = 1; i < member_->places_.size(); ++i) {
      member_->local_scopes_.emplace_back(&global_scope->NewScope());
    }
  } else {
    member_->own_local_scope_ = false;
    PADDLE_ENFORCE_EQ(member_->places_.size(),
                      local_scopes.size(),
                      common::errors::PreconditionNotMet(
                          "member_->places_.size() = %d is not equal to "
                          "local_scopes.size() = %d",
                          member_->places_.size(),
                          local_scopes.size()));
    for (size_t i = 0; i < member_->places_.size(); ++i) {
      if (create_new) {
        member_->local_scopes_.emplace_back(&local_scopes[i]->NewScope());
      } else {
        // Use local scopes directly
        member_->local_scopes_.emplace_back(local_scopes[i]);
      }
    }
  }
}

std::vector<ir::Graph *> CompiledProgram::CloneGraphToMultiDevices(
    ir::Graph *graph) {
  std::vector<ir::Graph *> graphs;
  if (member_->build_strategy_.async_mode_) {
    PADDLE_ENFORCE_EQ(member_->IsUseCUDA(member_->use_device_),
                      false,
                      common::errors::Unavailable(
                          "gpu mode does not support async_mode_ now!"));
    graphs.push_back(graph);
    for (size_t i = 1; i < member_->places_.size(); ++i) {
      auto *tmp_graph = new ir::Graph(graph->OriginProgram());
      graphs.push_back(tmp_graph);
    }
  }

  return graphs;
}

void CompiledProgram::PrepareNCCLCommunicator(Scope *global_scope) {
  if (member_->build_strategy_.reduce_ ==
      BuildStrategy::ReduceStrategy::kNoReduce) {
    return;
  }

  if (member_->IsUseCUDA(member_->use_device_) && member_->nranks_ > 1) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    member_->InitOrGetNCCLCommunicator(global_scope, &member_->build_strategy_);

    // Initialize device context's nccl comm, will be used by normal
    // Operators like sync_batch_norm, and collective ops.
    // NOTE: more than one CompiledProgram with same place, the nccl comm will
    // be rewrite and there will be some problem.
    // NOTE: NCCL group-calls and non-group-calls can not use the same
    // NCCL communicator, so for ParallelGraph and Multi-Process mode, re-use
    // same communicators.
    auto *nccl_ctxs = member_->nccl_ctxs_->GetSyncBatchNormCtx(
        global_scope, member_->places_);
    auto &pool = phi::DeviceContextPool::Instance();
    for (auto &place : member_->places_) {
      auto *dev_ctx = static_cast<phi::GPUContext *>(pool.Get(place));
      auto &nccl_ctx = nccl_ctxs->at(place);
      dev_ctx->set_nccl_comm(nccl_ctx.comm());
    }
#else
    PADDLE_THROW(common::errors::PreconditionNotMet("Not compiled with CUDA."));
#endif
  }
  if (member_->use_device_ == p::kXPU && member_->nranks_ > 1) {
#if defined(PADDLE_WITH_XPU_BKCL)
    member_->InitOrGetBKCLCommunicator(global_scope, member_->build_strategy_);

    auto *bkcl_ctxs = member_->bkcl_ctxs_->GetSyncBatchNormCtx(
        global_scope, member_->places_);
    auto &pool = phi::DeviceContextPool::Instance();
    for (size_t dev_id = 0; dev_id < member_->places_.size(); ++dev_id) {
      auto *dev_ctx =
          static_cast<phi::XPUContext *>(pool.Get(member_->places_[dev_id]));
      auto &bkcl_ctx = bkcl_ctxs->at(member_->places_[dev_id]);
      dev_ctx->SetBkclContext(bkcl_ctx.comm());
    }
#else
    PADDLE_THROW(common::errors::PreconditionNotMet("Not compiled with XPU."));
#endif
  }
}

std::vector<ir::Graph *> CompiledProgram::CompileGraphWithBuildStrategy(
    ir::Graph *graph,
    std::vector<ir::Graph *> *device_graphs,
    const std::string &loss_var_name) {
  auto device_count = member_->places_.size();
  std::vector<ir::Graph *> async_graphs(device_count);

  auto &graphs = *device_graphs;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  if (member_->build_strategy_.async_mode_) {
    PADDLE_ENFORCE_EQ(graphs.size(),
                      device_count,
                      common::errors::PreconditionNotMet(
                          "graphs.size() should be %d, but received %d",
                          device_count,
                          graphs.size()));
    VLOG(3) << "use local async mode";
    graph = member_->build_strategy_.Apply(graph,  // NOLINT
                                           {member_->places_[0]},
                                           loss_var_name,
                                           {member_->local_scopes_[0]},
                                           1,
                                           member_->use_device_,
                                           member_->nccl_ctxs_);
    for (size_t i = 1; i < device_count; ++i) {
      graphs[i] = member_->build_strategy_.Apply(graphs[i],
                                                 {member_->places_[i]},
                                                 loss_var_name,
                                                 {member_->local_scopes_[i]},
                                                 1,
                                                 member_->use_device_,
                                                 member_->nccl_ctxs_);
      async_graphs[i] = graphs[i];
    }
  } else {
    graph = member_->build_strategy_.Apply(graph,  // NOLINT
                                           member_->places_,
                                           loss_var_name,
                                           member_->local_scopes_,
                                           member_->nranks_,
                                           member_->use_device_,
                                           member_->nccl_ctxs_);
  }
#elif defined(PADDLE_WITH_XPU_BKCL)
  if (member_->build_strategy_.async_mode_) {
    PADDLE_ENFORCE_EQ(graphs.size(),
                      device_count,
                      common::errors::PreconditionNotMet(
                          "graphs.size() should be %d, but received %d",
                          device_count,
                          graphs.size()));
    VLOG(3) << "use local async mode";
    graph = member_->build_strategy_.Apply(graph,
                                           {member_->places_[0]},
                                           loss_var_name,
                                           {member_->local_scopes_[0]},
                                           1,
                                           member_->use_device_,
                                           member_->bkcl_ctxs_);
    for (size_t i = 1; i < device_count; ++i) {
      graphs[i] = member_->build_strategy_.Apply(graphs[i],
                                                 {member_->places_[i]},
                                                 loss_var_name,
                                                 {member_->local_scopes_[i]},
                                                 1,
                                                 member_->use_device_,
                                                 member_->bkcl_ctxs_);
      async_graphs[i] = graphs[i];
    }
  } else {
    graph = member_->build_strategy_.Apply(graph,
                                           member_->places_,
                                           loss_var_name,
                                           member_->local_scopes_,
                                           member_->nranks_,
                                           member_->use_device_,
                                           member_->bkcl_ctxs_);
  }
#else
  if (member_->build_strategy_.async_mode_) {
    VLOG(3) << "use local async mode";
    graph = member_->build_strategy_.Apply(graph,
                                           {member_->places_[0]},
                                           loss_var_name,
                                           {member_->local_scopes_[0]},
                                           1,
                                           member_->use_device_);
    for (size_t i = 1; i < device_count; ++i) {
      graphs[i] = member_->build_strategy_.Apply(graphs[i],
                                                 {member_->places_[i]},
                                                 loss_var_name,
                                                 {member_->local_scopes_[i]},
                                                 1,
                                                 member_->use_device_);
      async_graphs[i] = graphs[i];
    }
  } else {
    graph = member_->build_strategy_.Apply(graph,
                                           member_->places_,
                                           loss_var_name,
                                           member_->local_scopes_,
                                           member_->nranks_,
                                           member_->use_device_);
  }
#endif

  return async_graphs;
}

}  // namespace framework
}  // namespace paddle

USE_PASS(eager_deletion_pass);
