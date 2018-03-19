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
#include "ThreadPool.h"
#include "executor.h"
#include "lod_tensor.h"
#include "lod_tensor_array.h"
#include "op_registry.h"
#include "paddle/fluid/operators/math/concat.h"

namespace paddle {
namespace framework {

#ifdef PADDLE_WITH_CUDA

// FIXME: CHECK the return value of x;
#define NCCL_INVOKE(x) x
#endif

struct OpHandle;

struct VarHandleBase {
  virtual ~VarHandleBase() {}
  virtual std::string DebugString() const = 0;

  OpHandle *generated_op_;
  std::unordered_set<OpHandle *> pending_ops_;
};

struct VarHandle : public VarHandleBase {
  std::string DebugString() const override {
    std::stringstream ss;
    ss << name_ << ":" << place_;
    return ss.str();
  }

  // version field currently is not used, however, just store the version to
  // debug easily.
  size_t version_;
  std::string name_;
  platform::Place place_;
};

struct DummyVarHandle : public VarHandleBase {
  std::string DebugString() const override { return "dummy"; }
};

struct DependencyVarHandle : public VarHandleBase {
  std::string DebugString() const override { return "Dependency Variable"; }
};

struct OpHandle {
  std::vector<VarHandleBase *> inputs_;
  std::vector<VarHandleBase *> outputs_;
  std::unordered_map<platform::Place, platform::DeviceContext *,
                     platform::PlaceHash>
      dev_ctx_;

  std::string DebugString() {
    std::stringstream ss;
    ss << "(";
    for (auto *var : inputs_) {
      ss << var->DebugString() << ", ";
    }
    ss << ") --> (";
    for (auto *var : outputs_) {
      ss << var->DebugString() << ", ";
    }
    ss << ")\n";
    return ss.str();
  }

  virtual ~OpHandle() {}

  virtual void Run() { PADDLE_THROW("Not implemented"); }
  virtual void Wait(platform::DeviceContext *waited_dev) {}
};

struct ComputationOpHandle : public OpHandle {
  std::unique_ptr<OperatorBase> op_;
  Scope *scope_;
  platform::Place place_;

  explicit ComputationOpHandle(const OpDesc &op_desc, Scope *scope,
                               platform::Place place)
      : op_(framework::OpRegistry::CreateOp(op_desc)),
        scope_(scope),
        place_(place) {}

  void Run() override {
    // Wait other op if necessary
    if (platform::is_gpu_place(place_)) {
      int dev_id = boost::get<platform::CUDAPlace>(place_).device;
      cudaSetDevice(dev_id);
    }
    auto *cur_ctx = dev_ctx_[place_];
    for (auto *in : inputs_) {
      if (in->generated_op_ && in->generated_op_->dev_ctx_[place_] != cur_ctx) {
        in->generated_op_->Wait(cur_ctx);
      }
    }

    op_->Run(*scope_, place_);
  }

  void Wait(platform::DeviceContext *waited_dev) override {
    this->dev_ctx_.at(place_)->Wait();
  }
};

struct ScaleLossGradOpHandle : public OpHandle {
  float coeff_;
  Scope *scope_;
  platform::Place place_;
  cudaEvent_t ev_;

  explicit ScaleLossGradOpHandle(size_t num_dev, Scope *scope,
                                 platform::Place place)
      : coeff_(static_cast<float>(1.0 / num_dev)),
        scope_(scope),
        place_(place) {
    PADDLE_ENFORCE(cudaEventCreateWithFlags(&ev_, cudaEventDisableTiming));
  }

  ~ScaleLossGradOpHandle() { PADDLE_ENFORCE(cudaEventDestroy(ev_)); }

  void Run() override {
    std::string var_name = static_cast<VarHandle *>(this->outputs_[0])->name_;

    float *tmp = scope_->FindVar(var_name)
                     ->GetMutable<framework::LoDTensor>()
                     ->mutable_data<float>(make_ddim({1}), place_);

    if (platform::is_cpu_place(place_)) {
      *tmp = coeff_;
    } else {
      auto stream =
          static_cast<platform::CUDADeviceContext *>(this->dev_ctx_[place_])
              ->stream();
      cudaSetDevice(boost::get<platform::CUDAPlace>(place_).device);
      VLOG(3) << "1";
      PADDLE_ENFORCE(cudaGetLastError());
      VLOG(3) << "2";
      memory::Copy(boost::get<platform::CUDAPlace>(place_), tmp,
                   platform::CPUPlace(), &coeff_, sizeof(float), stream);
      PADDLE_ENFORCE(cudaGetLastError());
      VLOG(3) << "3";
      PADDLE_ENFORCE(cudaEventRecord(ev_, stream));
      VLOG(3) << "4";
    }
  }

  void Wait(platform::DeviceContext *waited_dev) override {
    if (platform::is_cpu_place(waited_dev->GetPlace())) {
      dev_ctx_.at(place_)->Wait();
    } else {
      auto stream =
          static_cast<platform::CUDADeviceContext *>(waited_dev)->stream();
      PADDLE_ENFORCE(cudaStreamWaitEvent(stream, ev_, 0));
    }
  }
};

struct FetchedData {
 public:
  std::vector<framework::LoDTensor> tensors_;

  explicit FetchedData(size_t num_fetched) { tensors_.resize(num_fetched); }
};

struct FetchOpHandle : public OpHandle {
  std::shared_ptr<FetchedData> data_;
  size_t offset_;
  std::vector<Scope *> *local_scopes_;
  std::vector<LoDTensor> tensors_;

  ~FetchOpHandle() {
    for (auto *input_var : inputs_) {
      input_var->pending_ops_.erase(this);
    }
    for (auto &pair : dev_ctx_) {
      pair.second->Wait();
    }

    // Lazily merge tensors. Will faster code.
    MergeTensors();
  }

  void Run() override {
    for (auto *input : inputs_) {
      auto *var = static_cast<VarHandle *>(input);
      var->generated_op_->Wait(this->dev_ctx_[var->place_]);
    }

    tensors_.resize(inputs_.size());
    auto *var = static_cast<VarHandle *>(inputs_[0]);
    auto &var_name = var->name_;
    platform::CPUPlace cpu;
    auto &scopes = *local_scopes_;

    for (size_t i = 0; i < scopes.size(); ++i) {
      auto &scope = scopes[i];
      auto &t = scope->FindVar(var_name)->Get<framework::LoDTensor>();
      if (platform::is_gpu_place(var->place_)) {
        TensorCopy(t, cpu, *dev_ctx_[t.place()], &tensors_[i]);
      } else {
        tensors_[i].ShareDataWith(t);
        tensors_[i].set_lod(t.lod());
      }
    }
  }

  void Wait(platform::DeviceContext *waited_dev) override {
    PADDLE_THROW("Nobody should wait FetchOp. Unexpceted Error");
  }

 private:
  void MergeTensors() const {
    std::vector<const LoDTensor *> tensors_ptr;
    for (auto &t : tensors_) {
      tensors_ptr.emplace_back(&t);
    }
    data_->tensors_[offset_].MergeLoDTensor(tensors_ptr, platform::CPUPlace());
  }
};

class ParallelExecutorPrivate {
 public:
  explicit ParallelExecutorPrivate(size_t num_threads = 12)
      : pool_(num_threads) {}

  std::vector<platform::Place> places_;

  std::vector<Scope *> local_scopes_;
  Scope *global_scope_;

#ifdef PADDLE_WITH_CUDA
  struct NCCLContext {
    std::unique_ptr<platform::CUDADeviceContext> ctx_;
    ncclComm_t comm;

    explicit NCCLContext(int dev_id) {
      ctx_.reset(new platform::CUDADeviceContext(platform::CUDAPlace(dev_id)));
    }

    cudaStream_t stream() const { return ctx_->stream(); }

    int device_id() const {
      return boost::get<platform::CUDAPlace>(ctx_->GetPlace()).device;
    }

    static void InitNCCLContext(std::unordered_map<int, NCCLContext> &contexts,
                                const std::vector<platform::Place> &places) {
      std::vector<ncclComm_t> comms;
      std::vector<int> devs;
      comms.resize(contexts.size());
      devs.reserve(contexts.size());

      for (auto &p : places) {
        devs.push_back(boost::get<platform::CUDAPlace>(p).device);
      }

      NCCL_INVOKE(platform::dynload::ncclCommInitAll(
          &comms[0], static_cast<int>(contexts.size()), &devs[0]));

      int i = 0;
      for (auto &dev_id : devs) {
        contexts.at(dev_id).comm = comms[i++];
      }
    }
  };

  std::unordered_map<int, NCCLContext> communication_streams_;

  NCCLContext &GetNCCLCtx(platform::Place p) {
    int dev_id = boost::get<platform::CUDAPlace>(p).device;
    return communication_streams_.at(dev_id);
  }

#endif

  platform::DeviceContext *CommunicationDevCtx(const platform::Place &place) {
    if (platform::is_cpu_place(place) || local_scopes_.size() == 1) {
      return const_cast<platform::DeviceContext *>(
          platform::DeviceContextPool::Instance().Get(place));
    } else {
#ifdef PADDLE_WITH_CUDA
      return GetNCCLCtx(place).ctx_.get();
#else
      PADDLE_THROW("Not compiled with CUDA")
#endif
    }
  }

  platform::Place main_place_;

  std::unordered_map<platform::Place,
                     std::unordered_map<std::string, std::map<int, VarHandle>>,
                     platform::PlaceHash>
      vars_;
  std::unordered_set<std::unique_ptr<VarHandleBase>> dep_vars_;

  std::vector<std::unique_ptr<OpHandle>> ops_;

  // Use a simpler thread pool, might be faster.
  ThreadPool pool_;

  std::unique_ptr<platform::EnforceNotMet> exception_;
};

// TODO(yy): Move this function somewhere
ncclDataType_t ToNCCLDataType(std::type_index type) {
  if (type == typeid(float)) {  // NOLINT
    return ncclFloat;
  } else if (type == typeid(double)) {  // NOLINT
    return ncclDouble;
  } else if (type == typeid(int)) {  // NOLINT
    return ncclInt;
  } else {
    PADDLE_THROW("Not supported");
  }
}

struct NCCLAllReduceOpHandle : public OpHandle {
  ParallelExecutorPrivate *member_;
  std::unordered_map<int, cudaEvent_t> events_;

  explicit NCCLAllReduceOpHandle(ParallelExecutorPrivate *member)
      : member_(member) {
    for (auto &nccl : member_->communication_streams_) {
      PADDLE_ENFORCE(cudaEventCreate(&events_[nccl.second.device_id()],
                                     cudaEventDisableTiming));
    }
  }

  ~NCCLAllReduceOpHandle() {
    for (auto &ev : events_) {
      PADDLE_ENFORCE(cudaEventDestroy(ev.second));
    }
  }

  void Run() override {
    if (this->inputs_.size() == 1) {
      return;  // No need to all reduce when GPU count = 1;
    } else {
      auto &var_name = static_cast<VarHandle *>(this->inputs_[0])->name_;

      int dtype = -1;
      size_t numel = 0;

      platform::dynload::ncclGroupStart();

      for (size_t i = 0; i < member_->local_scopes_.size(); ++i) {
        auto &p = member_->places_[i];
        auto *s = member_->local_scopes_[i];
        int dev_id = boost::get<platform::CUDAPlace>(p).device;

        auto &lod_tensor = s->FindVar(var_name)->Get<framework::LoDTensor>();
        void *buffer = const_cast<void *>(lod_tensor.data<void>());
        if (dtype == -1) {
          dtype = ToNCCLDataType(lod_tensor.type());
        }

        if (numel == 0) {
          numel = static_cast<size_t>(lod_tensor.numel());
        }

        auto &nccl_ctx = member_->communication_streams_.at(dev_id);
        cudaSetDevice(dev_id);
        platform::dynload::ncclAllReduce(
            buffer, buffer, numel, static_cast<ncclDataType_t>(dtype), ncclSum,
            nccl_ctx.comm, nccl_ctx.stream());
        PADDLE_ENFORCE(cudaEventRecord(events_[dev_id], nccl_ctx.stream()));
      }

      platform::dynload::ncclGroupEnd();
    }
  }

  void Wait(platform::DeviceContext *waited_dev) override {
    if (platform::is_cpu_place(
            waited_dev->GetPlace())) {  // Wait by CPU, just sync stream
      for (auto &pair : member_->communication_streams_) {
        pair.second.ctx_->Wait();
      }
    } else {
      if (events_.size() > 1) {
        int dev_id =
            boost::get<platform::CUDAPlace>(waited_dev->GetPlace()).device;
        auto stream =
            static_cast<platform::CUDADeviceContext *>(waited_dev)->stream();
        PADDLE_ENFORCE(cudaStreamWaitEvent(stream, events_[dev_id], 0));
      }
    }
  }
};

ParallelExecutor::ParallelExecutor(
    const std::vector<platform::Place> &places,
    const std::unordered_set<std::string> &params,
    const ProgramDesc &startup_program, const ProgramDesc &main_program,
    const std::string &loss_var_name, Scope *scope)
    : member_(new ParallelExecutorPrivate()) {
  member_->places_ = places;
  member_->global_scope_ = scope;
  // Step 1. RunStartupProgram and Bcast the params to devs.
  Executor exe(places[0]);
  exe.Run(startup_program, scope, 0);
  // Create local scopes
  for (size_t i = 0; i < member_->places_.size(); ++i) {
    member_->local_scopes_.push_back(&scope->NewScope());
  }
  member_->main_place_ = places[0];

  // Bcast Parameters to all GPUs
  BuildNCCLCommunicator();
  if (platform::is_gpu_place(member_->main_place_) &&
      member_->local_scopes_.size() != 1) {  // Is CUDA
    BCastParamsToGPUs(startup_program);
  }
  // Startup Program has been run. All local scopes has correct parameters.

  // Step 2. Convert main_program to SSA form and dependency graph. Also, insert
  // ncclOp
  ConstructDependencyGraph(params, main_program, loss_var_name);

  // Step 3. Create vars in each scope;
  for (auto *scope : member_->local_scopes_) {
    for (auto *var : main_program.Block(0).AllVars()) {
      if (scope->FindVar(var->Name()) != nullptr) {
        continue;
      }

      InitializeVariable(scope->Var(var->Name()), var->GetType());
    }
  }
}

void ParallelExecutor::ConstructDependencyGraph(
    const std::unordered_set<std::string> &params,
    const ProgramDesc &main_program, const std::string &loss_var_name) const {
  std::unordered_set<std::string> grads;
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

    for (size_t i = 0; i < member_->places_.size(); ++i) {
      auto &p = member_->places_[i];
      auto *s = member_->local_scopes_[i];

      member_->ops_.emplace_back(new ComputationOpHandle(*op, s, p));
      auto *op_handle = member_->ops_.back().get();
      op_handle->dev_ctx_[p] = const_cast<platform::DeviceContext *>(
          platform::DeviceContextPool::Instance().Get(p));

      auto var_names = op->InputArgumentNames();

      for (auto &each_var_name : var_names) {
        VarHandle *var = GetVarHandle(each_var_name, p);
        op_handle->inputs_.emplace_back(var);
        var->pending_ops_.emplace(op_handle);
      }
      var_names = op->OutputArgumentNames();

      for (auto &each_var_name : var_names) {
        GenerateVar(op_handle, each_var_name, p);
      }

      if (is_forwarding) {
        if (var_names.size() == 1 && var_names[0] == loss_var_name) {
          // Insert ScaleCost OpHandle
          member_->ops_.emplace_back(new ScaleLossGradOpHandle(
              this->member_->local_scopes_.size(), s, p));
          op_handle = member_->ops_.back().get();

          op_handle->dev_ctx_[p] = member_->CommunicationDevCtx(p);

          // FIXME: Currently ScaleLossGradOp only use device_count as scale
          // factor. So it does not depend on any other operators.
          // VarHandle *loss = GetVarHandle(loss_var_name, place);
          // loss->pending_ops_.emplace_back(op_handle);
          // op_handle->inputs_.emplace_back(loss);

          GenerateVar(op_handle, loss_var_name + "@GRAD", p);
          change_forward = true;
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
          member_->ops_.emplace_back(new NCCLAllReduceOpHandle(member_));
          auto *op_handle = member_->ops_.back().get();

          for (size_t i = 0; i < member_->places_.size(); ++i) {
            auto &p = member_->places_[i];
            auto &vars = member_->vars_[p][og];

            if (vars.empty()) {  // This device has no data. continue.
              continue;
            }
            auto *prev_grad = &vars[vars.size() - 1];
            op_handle->inputs_.emplace_back(prev_grad);
            prev_grad->pending_ops_.emplace(op_handle);
            auto &var = vars[vars.size()];
            var.place_ = p;
            var.generated_op_ = op_handle;
            var.name_ = og;
            var.version_ = vars.size() - 1;
            op_handle->outputs_.emplace_back(&var);
            op_handle->dev_ctx_[p] = member_->CommunicationDevCtx(p);
          }
        }
      }
    }
  }

  /*
    Dependency graph has been constructed. However, there are still data
    harzaeds need to be handled.
   */
  PolishGraphToSupportDataHarzaeds();
}

/**
 * We only handle write after read(WAR), since it should not have a write
 * after write in program. If there are write after write operators, we need
 * prune them.
 *
 * https://en.wikipedia.org/wiki/Hazard_(computer_architecture)#Write_after_read_(WAR)
 */
void ParallelExecutor::PolishGraphToSupportDataHarzaeds() const {
  for (auto &place_pair : member_->vars_) {
    for (auto &name_pair : place_pair.second) {
      if (name_pair.second.size() <= 1) {
        return;
      }
      auto it_new = name_pair.second.rbegin();
      auto it_old = name_pair.second.rbegin();
      ++it_old;
      for (; it_old != name_pair.second.rend(); it_new = it_old, ++it_old) {
        auto *write_op = it_new->second.generated_op_;
        auto &read_ops = it_old->second.pending_ops_;
        auto *ex_write_op = it_old->second.generated_op_;

        if (ex_write_op == nullptr) {  // Nobody write this var.
          continue;
        }

        for (auto *read_op : read_ops) {
          // Manually add a dependency var from read_op to write_op;
          if (read_op == write_op) {
            // Read Write is the same op.
            continue;
          }

          auto *dep_var = new DependencyVarHandle();

          dep_var->generated_op_ = read_op;
          read_op->outputs_.emplace_back(dep_var);

          dep_var->pending_ops_.emplace(write_op);
          write_op->inputs_.emplace_back(dep_var);
          member_->dep_vars_.emplace(dep_var);
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
#ifdef PADDLE_WITH_CUDA
  auto *main_scope = member_->local_scopes_[0];

  for (auto *var_desc : startup_program.Block(0).AllVars()) {
    if (var_desc->GetType() == proto::VarType::LOD_TENSOR) {
      auto &main_tensor =
          main_scope->FindVar(var_desc->Name())->Get<LoDTensor>();
      ncclDataType_t data_type = ToNCCLDataType(main_tensor.type());
      auto &dims = main_tensor.dims();
      size_t numel = main_tensor.numel();

      platform::dynload::ncclGroupStart();

      for (size_t i = 0; i < member_->places_.size(); ++i) {
        auto place = member_->places_[i];
        void *buffer;
        if (i == 0) {
          buffer = const_cast<void *>(main_tensor.data<void>());
        } else {
          auto local_scope = member_->local_scopes_[i];
          auto *t = local_scope->Var(var_desc->Name())->GetMutable<LoDTensor>();
          t->Resize(dims);
          buffer = t->mutable_data(place, main_tensor.type());
        }

        auto &nccl_ctx = member_->GetNCCLCtx(place);
        platform::dynload::ncclBcast(buffer, numel, data_type, 0, nccl_ctx.comm,
                                     nccl_ctx.stream());
      }
      platform::dynload::ncclGroupEnd();
    }
  }
#else
  PADDLE_THROW("Not compiled with CUDA");
#endif
}

void ParallelExecutor::BuildNCCLCommunicator() const {
#ifdef PADDLE_WITH_CUDA
  for (auto &place : member_->places_) {
    int dev_id = boost::get<platform::CUDAPlace>(place).device;

    member_->communication_streams_.emplace(
        dev_id, ParallelExecutorPrivate::NCCLContext(dev_id));
  }

  ParallelExecutorPrivate::NCCLContext::InitNCCLContext(
      member_->communication_streams_, member_->places_);
#endif
}

void ParallelExecutor::Run(const std::vector<std::string> &fetch_tensors,
                           const std::string &fetched_var_name) {
  auto fetched_data = std::make_shared<FetchedData>(fetch_tensors.size());
  // Version --> VarHandle
  member_->exception_.reset();
  std::unordered_map<VarHandleBase *, std::atomic<bool>> pending_vars;
  std::unordered_map<OpHandle *, size_t> pending_ops;
  std::vector<DummyVarHandle> dummy_vars;

  for (auto &place_pair : member_->vars_) {
    for (auto &name_pair : place_pair.second) {
      for (auto &version_pair : name_pair.second) {
        pending_vars[&version_pair.second] =
            version_pair.second.generated_op_ == nullptr;
      }
    }
  }

  for (auto &var : member_->dep_vars_) {
    pending_vars[var.get()] = var->generated_op_ == nullptr;
  }

  std::vector<OpHandle *> to_run;

  for (auto &op : member_->ops_) {
    if (op->inputs_.empty()) {  // Special case, Op has no input.
      to_run.emplace_back(op.get());
    } else {
      pending_ops.insert({op.get(), op->inputs_.size()});
    }
  }

  std::unordered_map<std::string, std::vector<VarHandleBase *>> fetched_vars;

  for (auto &fetch_var_name : fetch_tensors) {
    for (auto &pair : member_->vars_) {
      auto it = pair.second.find(fetch_var_name);
      if (it != pair.second.end()) {
        fetched_vars[fetch_var_name].push_back(&it->second.rbegin()->second);
      }
    }
  }

  std::vector<FetchOpHandle> fetch_ops;

  for (size_t i = 0; i < fetch_tensors.size(); ++i) {
    auto &var_name = fetch_tensors[i];
    auto &vars = fetched_vars[var_name];
    fetch_ops.emplace_back();
    FetchOpHandle *op = &fetch_ops.back();
    op->data_ = fetched_data;
    op->offset_ = i;
    op->local_scopes_ = &member_->local_scopes_;
    for (auto &p : member_->places_) {
      op->dev_ctx_[p] = member_->GetNCCLCtx(p).ctx_.get();
    }

    for (auto *var : vars) {
      var->pending_ops_.emplace(op);
      op->inputs_.emplace_back(var);
    }

    dummy_vars.emplace_back();
    auto *var = &dummy_vars.back();
    op->outputs_.emplace_back(var);
    var->generated_op_ = op;
    pending_vars[var] = false;

    pending_ops.insert({op, op->inputs_.size()});
  }

  for (auto *op : to_run) {
    RunOp(pending_vars, op);
  }

  while (!pending_vars.empty()) {
    VarHandleBase *ready_var = nullptr;
    for (auto &pair : pending_vars) {
      if (pair.second.load(std::memory_order_consume)) {
        ready_var = pair.first;
      }
    }
    if (ready_var == nullptr) {
      // FIXME use conditional var instead of busy wait.
      if (member_->exception_) {
        throw * member_->exception_;
      }
      continue;
    }
    pending_vars.erase(ready_var);
    to_run.clear();
    for (auto *op : ready_var->pending_ops_) {
      auto &deps = pending_ops[op];
      --deps;
      if (deps == 0) {
        to_run.emplace_back(op);
      }
    }
    for (auto *op : to_run) {
      pending_ops.erase(op);
      RunOp(pending_vars, op);
    }
  }

  fetch_ops.clear();
  *member_->global_scope_->Var(fetched_var_name)->GetMutable<LoDTensorArray>() =
      fetched_data->tensors_;
}

void ParallelExecutor::RunOp(
    std::unordered_map<VarHandleBase *, std::atomic<bool>> &pending_vars,
    OpHandle *op) const {
  std::vector<std::atomic<bool> *> *ready_buffer =
      new std::vector<std::atomic<bool> *>();
  for (auto *var : op->outputs_) {
    ready_buffer->emplace_back(&pending_vars[var]);
  }

  auto op_run = [ready_buffer, op, this] {
    try {
      VLOG(10) << op->DebugString();
      op->Run();
      for (auto *ready : *ready_buffer) {
        ready->store(true, std::memory_order_release);
      }
      delete ready_buffer;
    } catch (platform::EnforceNotMet ex) {
      member_->exception_.reset(new platform::EnforceNotMet(ex));
    } catch (...) {
      LOG(FATAL) << "Unknown exception catched";
    }
  };
  member_->pool_.enqueue(op_run);
}
}  // namespace framework
}  // namespace paddle
