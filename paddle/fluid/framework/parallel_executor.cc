/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/parallel_executor.h"
#include "lod_tensor.h"
#include "op_registry.h"
#include "threadpool.h"

namespace paddle {
namespace framework {

struct OpHandle;

struct VarHandle {
  size_t version_;
  std::string name_;
  platform::Place place_;

  OpHandle *generated_op_;

  std::vector<OpHandle *> pending_ops_;
};

struct OpHandle {
  std::vector<VarHandle *> inputs_;
  std::vector<VarHandle *> outputs_;

  std::string DebugString() {
    std::stringstream ss;
    ss << "(";
    for (auto *var : inputs_) {
      ss << var->name_ << ":" << var->place_ << ", ";
    }
    ss << ") --> (";
    for (auto *var : outputs_) {
      ss << var->name_ << ":" << var->place_ << ", ";
    }
    ss << ")\n";
    return ss.str();
  }

  virtual ~OpHandle() {}
};

struct ComputationOpHandle : public OpHandle {
  std::unique_ptr<OperatorBase> op_;

  explicit ComputationOpHandle(const OpDesc &op_desc)
      : op_(framework::OpRegistry::CreateOp(op_desc)) {}
};

struct ScaleLossGradOpHandle : public OpHandle {};

struct NCCLAllReduceOpHandle : public OpHandle {};

class ParallelExecutorPrivate {
 public:
  explicit ParallelExecutorPrivate(size_t num_threads = 12)
      : pool_(num_threads) {}

  std::unordered_map<platform::Place, Scope *, platform::PlaceHash>
      local_scopes_;
  std::unordered_map<platform::Place, platform::CUDADeviceContext,
                     platform::PlaceHash>
      dev_ctxs_;
  platform::Place main_place_;

  std::unordered_map<platform::Place,
                     std::unordered_map<std::string, std::map<int, VarHandle>>,
                     platform::PlaceHash>
      vars_;
  std::vector<std::unique_ptr<OpHandle>> ops_;

  ThreadPool pool_;
};

// TODO(yy): Move this function somewhere
ncclDataType_t ToNCCLDataType(std::type_index type) {
  // FIXME!!
  return ncclFloat;
}

ParallelExecutor::ParallelExecutor(
    const std::vector<platform::Place> &places,
    const std::unordered_set<std::string> &params,
    const ProgramDesc &startup_program, const ProgramDesc &main_program,
    const std::string &loss_var_name, Scope *scope)
    : member_(new ParallelExecutorPrivate()) {
  // Step 1. RunStartupProgram and Bcast the params to devs.
  Executor exe(places[0]);
  exe.Run(startup_program, scope, 0);
  // Create local scopes
  for (auto &place : places) {
    member_->local_scopes_[place] = &scope->NewScope();
  }
  member_->main_place_ = places[0];

  // Bcast Parameters to all GPUs
  if (platform::is_gpu_place(member_->main_place_)) {  // Is CUDA
    //    BCastParamsToGPUs(startup_program);
  }
  // Startup Program has been run. All local scopes has correct parameters.

  // Step 2. Convert main_program to SSA form and dependency graph. Also, insert
  // ncclOp
  ConstructDependencyGraph(params, main_program, loss_var_name);
}

void ParallelExecutor::ConstructDependencyGraph(
    const std::unordered_set<std::string> &params,
    const ProgramDesc &main_program, const std::string &loss_var_name) const {
  std::unordered_set<std::__cxx11::string> grads;
  for (auto &each_param : params) {
    grads.insert(each_param + "@GRAD");
  }

  bool is_forwarding = true;
  for (auto *op : main_program.Block(0).AllOps()) {
    bool change_forward = false;

    if (!is_forwarding) {
      // FIXME(yy): Do not hard code like this
      if (op->OutputArgumentNames().size() == 1 &&
          op->OutputArgumentNames()[0] == loss_var_name + "@GRAD") {
        continue;  // Drop fill 1. for backward coeff;
      }
    }

    for (auto &pair : member_->local_scopes_) {
      member_->ops_.emplace_back(new ComputationOpHandle(*op));
      auto *op_handle = member_->ops_.back().get();

      auto var_names = op->InputArgumentNames();

      for (auto &each_var_name : var_names) {
        auto &place = pair.first;
        VarHandle *var = GetVarHandle(each_var_name, place);
        op_handle->inputs_.emplace_back(var);
        var->pending_ops_.emplace_back(op_handle);
      }
      var_names = op->OutputArgumentNames();

      for (auto &each_var_name : var_names) {
        auto &place = pair.first;
        GenerateVar(op_handle, each_var_name, place);
      }

      if (is_forwarding) {
        if (var_names.size() == 1 && var_names[0] == loss_var_name) {
          // Insert ScaleCost OpHandle
          member_->ops_.emplace_back(new ScaleLossGradOpHandle());

          op_handle = member_->ops_.back().get();
          auto &place = pair.first;
          VarHandle *loss = GetVarHandle(loss_var_name, place);
          loss->pending_ops_.emplace_back(op_handle);
          op_handle->inputs_.emplace_back(loss);
          GenerateVar(op_handle, loss_var_name + "@GRAD", place);
          change_forward = true;
          LOG(INFO) << "Scale Loss " << op_handle->DebugString();
        }
      }
    }

    if (change_forward) {
      is_forwarding = false;
    }

    if (!is_forwarding) {
      auto var_names = op->OutputArgumentNames();
      for (auto &og : var_names) {
        if (grads.count(og) != 0) {  // is param grad
          // Insert NCCL AllReduce Op
          member_->ops_.emplace_back(new NCCLAllReduceOpHandle());
          auto *op_handle = member_->ops_.back().get();

          for (auto &pair : member_->local_scopes_) {
            auto &place = pair.first;
            auto &vars = member_->vars_[place][og];

            if (vars.empty()) {  // This device has no data. continue.
              continue;
            }
            auto *prev_grad = &vars[vars.size() - 1];
            op_handle->inputs_.emplace_back(prev_grad);
            prev_grad->pending_ops_.emplace_back(op_handle);
            auto &var = vars[vars.size()];
            var.place_ = place;
            var.generated_op_ = op_handle;
            var.name_ = og;
            var.version_ = vars.size() - 1;
            op_handle->outputs_.emplace_back(&var);
          }
        }
      }
    }
  }
}

void ParallelExecutor::GenerateVar(OpHandle *op_handle,
                                   const std::string &each_var_name,
                                   const platform::Place &place) const {
  auto &vars = member_->vars_[place][each_var_name];
  size_t version = vars.size();
  auto &var = vars[version];
  var.version_ = version;
  var.generated_op_ = op_handle;
  var.name_ = each_var_name;
  var.place_ = place;
  op_handle->outputs_.emplace_back(&var);
}

VarHandle *ParallelExecutor::GetVarHandle(const std::string &each_var_name,
                                          const platform::Place &place) const {
  auto &var_holders = member_->vars_[place];
  auto &var_holder = var_holders[each_var_name];
  VarHandle *var = nullptr;
  if (var_holder.empty()) {
    auto &init_var = var_holder[0];
    init_var.place_ = place;
    init_var.name_ = each_var_name;
    init_var.generated_op_ = nullptr;
    init_var.version_ = 0;
    var = &init_var;
  } else {
    var = &var_holder.rbegin()->second;
  }
  return var;
}

void ParallelExecutor::BCastParamsToGPUs(
    const ProgramDesc &startup_program) const {
  auto *main_scope = member_->local_scopes_[member_->main_place_];
  for (auto *var_desc : startup_program.Block(0).AllVars()) {
    if (var_desc->GetType() == proto::VarType::LOD_TENSOR) {
      auto &main_tensor =
          main_scope->FindVar(var_desc->Name())->Get<LoDTensor>();

      ncclDataType_t data_type = ToNCCLDataType(main_tensor.type());
      auto &dims = main_tensor.dims();
      size_t numel = main_tensor.numel();
      std::vector<std::pair<void *, const platform::DeviceContext *>> mems;
      mems.emplace_back(
          const_cast<void *>(main_tensor.data<void>()),
          new platform::CUDADeviceContext(
              boost::get<platform::CUDAPlace>(member_->main_place_)));

      for (auto &pair : member_->local_scopes_) {
        if (pair.first == member_->main_place_) {
          continue;
        }

        auto local_scope = pair.second;
        auto *t = local_scope->Var(var_desc->Name())->GetMutable<LoDTensor>();
        t->Resize(dims);
        mems.emplace_back(t->mutable_data(pair.first, main_tensor.type()),
                          new platform::CUDADeviceContext(
                              boost::get<platform::CUDAPlace>(pair.first)));
      }

      // TODO(yy): Invoke ncclBCast here. mems, numel, data_type. The mems[0]
      // is the src, rests are dests.

      (void)(data_type);
      (void)(numel);

      // Free Communication Ctx
      for (auto &pair : mems) {
        // Release Communication Ctx

        // FIXME: Store CUDA DevCtx to member. Since NCCL All Reduce will use
        // this
        delete pair.second;
      }
    }
  }
}

std::vector<LoDTensor> ParallelExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  // Version --> VarHandle

  std::unordered_map<VarHandle *, bool> pending_vars;
  std::unordered_map<OpHandle *, size_t> pending_ops;

  for (auto &place_pair : member_->vars_) {
    for (auto &name_pair : place_pair.second) {
      for (auto &version_pair : name_pair.second) {
        pending_vars[&version_pair.second] =
            version_pair.second.generated_op_ == nullptr;
      }
    }
  }

  for (auto &op : member_->ops_) {
    pending_ops.insert({op.get(), op->inputs_.size()});
  }

  while (!pending_ops.empty()) {
    VarHandle *ready_var = nullptr;
    for (auto &pair : pending_vars) {
      if (pair.second) {
        ready_var = pair.first;
      }
    }

    if (ready_var == nullptr) {
      member_->pool_.Wait();  // Wait thread pool;
      continue;
    }

    pending_vars.erase(ready_var);

    std::vector<OpHandle *> to_run;

    for (auto *op : ready_var->pending_ops_) {
      auto &deps = pending_ops[op];
      --deps;
      if (deps == 0) {
        to_run.emplace_back(op);
      }
    }

    for (auto *op : to_run) {
      pending_ops.erase(op);

      std::vector<bool *> ready_buffer;
      for (auto *var : op->outputs_) {
        ready_buffer.emplace_back(&pending_vars[var]);
      }

      auto op_run = [ready_buffer, op] {
        // TODO(yy) Check Previous Op has same dev ctx.
        LOG(INFO) << "Run " << op->DebugString();
        for (auto *ready : ready_buffer) {
          *ready = true;
        }
      };

      member_->pool_.Run(op_run);
    }
  }
  return std::vector<LoDTensor>();
}
}  // namespace framework
}  // namespace paddle
