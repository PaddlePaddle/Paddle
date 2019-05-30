// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/layer.h"
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace imperative {

template <typename T>
class TensorAddFunctor : public boost::static_visitor<> {
 public:
  TensorAddFunctor(int64_t numel, const T* x, T* y)
      : numel_(numel), x_(x), y_(y) {}

  void operator()(const platform::CPUPlace& place) {
    platform::CPUDeviceContext* ctx = dynamic_cast<platform::CPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(place));
    auto blas = operators::math::GetBlas<platform::CPUDeviceContext, T>(*ctx);
    blas.AXPY(numel_, 1., x_, y_);
  }

#ifdef PADDLE_WITH_CUDA
  void operator()(const platform::CUDAPlace& place) {
    platform::CUDADeviceContext* ctx =
        dynamic_cast<platform::CUDADeviceContext*>(
            platform::DeviceContextPool::Instance().Get(place));
    auto blas = operators::math::GetBlas<platform::CUDADeviceContext, T>(*ctx);
    blas.AXPY(numel_, 1., x_, y_);
  }
#else
  void operator()(const platform::CUDAPlace& place) {
    PADDLE_THROW("Do NOT support gradient merge in place %s", place);
  }
#endif

  // there is NO blas in CUDAPinnedPlace
  void operator()(const platform::CUDAPinnedPlace& place) {
    PADDLE_THROW("Do NOT support gradient merge in place %s", place);
  }

 private:
  int64_t numel_;
  const T* x_;
  T* y_;
};

static void TensorAdd(const framework::Variable& src,
                      framework::Variable* dst) {
  auto* dst_tensor = dst->GetMutable<framework::LoDTensor>();
  auto& src_tensor = src.Get<framework::LoDTensor>();

  auto numel = src_tensor.numel();

  // FIXME(minqiyang): loss_grad op will pass a zero grad of label
  // ugly fix for it
  if (numel == 0) {
    return;
  }

  PADDLE_ENFORCE(dst_tensor->numel() == numel, "dst_numel %d vs. src_numel %d",
                 dst_tensor->numel(), numel);

  auto data_type = src_tensor.type();
  auto place = src_tensor.place();

#define TENSOR_ADD_MACRO(cpp_type)                                 \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType) { \
    TensorAddFunctor<cpp_type> func(                               \
        numel, src_tensor.data<cpp_type>(),                        \
        dst_tensor->mutable_data<cpp_type>(place));                \
    boost::apply_visitor(func, place);                             \
    return;                                                        \
  }

  TENSOR_ADD_MACRO(float);
  TENSOR_ADD_MACRO(double);

#undef TENSOR_ADD_MACRO

  PADDLE_THROW("Not supported data type %s for AddTo",
               framework::DataTypeToString(data_type));
}

template <bool kIsInput>
static framework::VariableNameMap CreateVarNameMap(
    const framework::OpInfo* op_info, const std::string& op_type,
    const NameVarBaseMap& varbase_map) {
  if (op_info == nullptr || op_info->proto_ == nullptr) {
    return {};
  }

  VLOG(2) << "CreateVarNameMap " << kIsInput;
  framework::VariableNameMap result;

  for (auto& var :
       kIsInput ? op_info->Proto().inputs() : op_info->Proto().outputs()) {
    auto it = varbase_map.find(var.name());
    if (it == varbase_map.end()) {
      PADDLE_ENFORCE(var.dispensable());
      result[var.name()] = {};
    } else {
      auto& var_vector = it->second;
      std::vector<std::string> args;
      args.reserve(var_vector.size());
      for (auto& var_base : var_vector) {
        args.emplace_back(var_base->Name());
      }
      result[var.name()] = std::move(args);
    }
  }

  VLOG(2) << "CreateVarNameMap " << kIsInput << " done";
  return result;
}

static framework::RuntimeContext PrepareRuntimeContext(
    const NameVarBaseMap& ins, const NameVarBaseMap& outs) {
  framework::VariableValueMap inputs, outputs;
  for (auto& in_pair : ins) {
    auto& in_ctx = inputs[in_pair.first];
    in_ctx.reserve(in_pair.second.size());
    for (auto& in_var : in_pair.second) {
      in_ctx.emplace_back(in_var->MutableVar());
    }
  }

  for (auto& out_pair : outs) {
    auto& out_ctx = outputs[out_pair.first];
    out_ctx.reserve(out_pair.second.size());
    for (auto& out_var : out_pair.second) {
      out_ctx.emplace_back(out_var->MutableVar());
    }
  }
  return framework::RuntimeContext(std::move(inputs), std::move(outputs));
}

class PreparedOp {
 private:
  PreparedOp(const framework::OperatorBase& op,
             const framework::RuntimeContext& ctx,
             framework::OperatorWithKernel::OpKernelFunc func,
             platform::DeviceContext* dev_ctx,
             std::vector<framework::KernelConfig>* kernel_configs)
      : op_(op),
        ctx_(ctx),
        func_(std::move(func)),
        dev_ctx_(dev_ctx),
        kernel_configs_(kernel_configs) {}

  static const framework::Tensor* GetTensorFromVar(
      const framework::Variable& var) {
    if (var.IsType<framework::LoDTensor>()) {
      return &(var.Get<framework::LoDTensor>());
    } else if (var.IsType<framework::SelectedRows>()) {
      return &(var.Get<framework::SelectedRows>().value());
    } else {
      return nullptr;
    }
  }

 public:
  static platform::Place GetExpectedPlace(const platform::Place& place,
                                          const NameVarBaseMap& ins) {
    bool found = false;
    for (auto& name_pair : ins) {
      for (auto& var_base : name_pair.second) {
        const auto* tensor = GetTensorFromVar(var_base->Var());
        if (tensor && tensor->IsInitialized()) {
          auto tmp_place = tensor->place();
          PADDLE_ENFORCE(!found || tmp_place == place,
                         "Input variable should keep in the same place: %s, "
                         "but get place: %s of input %s instead",
                         place, tmp_place, name_pair.first);
        }
      }
    }
    return place;
  }

  static PreparedOp Prepare(const framework::RuntimeContext& ctx,
                            const framework::OperatorWithKernel& op,
                            const platform::Place& place) {
    auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);

    // check if op[type] has kernel registered.
    auto& all_op_kernels = op.AllOpKernels();
    auto kernels_iter = all_op_kernels.find(op.Type());
    if (kernels_iter == all_op_kernels.end()) {
      PADDLE_THROW(
          "There are no kernels which are registered in the %s operator.",
          op.Type());
    }

    auto& kernels = kernels_iter->second;

    auto expected_kernel_key =
        op.GetExpectedKernelType(framework::ExecutionContext(
            op, framework::Scope(), *dev_ctx, ctx, nullptr));
    VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

    auto kernel_iter = kernels.find(expected_kernel_key);
#ifdef PADDLE_WITH_MKLDNN
    // workaround for missing MKLDNN kernel when FLAGS_use_mkldnn env var is set
    if (kernel_iter == kernels.end() &&
        expected_kernel_key.library_type_ == framework::LibraryType::kMKLDNN) {
      VLOG(3) << "missing MKLDNN kernel: fallbacking to PLAIN one";
      expected_kernel_key.library_type_ = framework::LibraryType::kPlain;
      expected_kernel_key.data_layout_ = framework::DataLayout::kAnyLayout;
      kernel_iter = kernels.find(expected_kernel_key);
    }
#endif
    if (kernel_iter == kernels.end()) {
      PADDLE_THROW("op %s does not have kernel for %s", op.Type(),
                   KernelTypeToString(expected_kernel_key));
    }
    std::vector<framework::KernelConfig>* kernel_configs =
        op.GetKernelConfig(expected_kernel_key);
    return PreparedOp(op, ctx, kernel_iter->second, dev_ctx, kernel_configs);
  }

  inline platform::DeviceContext* GetDeviceContext() const { return dev_ctx_; }

  void Run() {
    // TODO(zjl): remove scope in dygraph
    framework::Scope scope;
    op_.RuntimeInferShape(scope, dev_ctx_->GetPlace(), ctx_);
    func_(framework::ExecutionContext(op_, scope, *dev_ctx_, ctx_,
                                      kernel_configs_));
  }

 private:
  const framework::OperatorBase& op_;
  const framework::RuntimeContext& ctx_;
  framework::OperatorWithKernel::OpKernelFunc func_;
  platform::DeviceContext* dev_ctx_;
  std::vector<framework::KernelConfig>* kernel_configs_;
};

static std::vector<std::unique_ptr<framework::OpDesc>> CreateGradOpDescs(
    const framework::OpDesc& op_desc,
    const std::unordered_set<std::string>& no_grad_set,
    const std::vector<framework::BlockDesc*>& grad_sub_block,
    std::unordered_map<std::string, std::string>* grad_to_var) {
  auto& op_info = framework::OpInfoMap::Instance().Get(op_desc.Type());
  if (!op_info.grad_op_maker_) return {};
  return op_info.GradOpMaker()(op_desc, no_grad_set, grad_to_var,
                               grad_sub_block);
}

void Tracer::TraceOp(const std::string& type, const NameVarBaseMap& ins,
                     const NameVarBaseMap& outs, framework::AttributeMap attrs,
                     const platform::Place& place, bool trace_backward) {
  size_t op_id = GenerateUniqueId();
  auto op = OpBase::Create(this, op_id, type, ins, outs, std::move(attrs),
                           place, trace_backward);
  if (trace_backward) {
    auto* op_ptr = op.get();
    ops_.emplace(op_id, std::move(op));
    for (auto& pair : outs) {
      for (auto& var : pair.second) {
        var->SetGeneratedOp(op_ptr);
      }
    }
  }
}

void Tracer::TraceOp(const framework::OpDesc& op_desc,
                     const NameVarBaseMap& ins, const NameVarBaseMap& outs,
                     const platform::Place& place, bool trace_backward) {
  size_t op_id = GenerateUniqueId();
  auto op =
      OpBase::Create(this, op_id, op_desc, ins, outs, place, trace_backward);

  if (trace_backward) {
    auto* op_ptr = op.get();
    ops_.emplace(op_id, std::move(op));
    for (auto& pair : outs) {
      for (auto& var : pair.second) {
        var->SetGeneratedOp(op_ptr);
      }
    }
  }
}

void VarBase::SetGeneratedOp(OpBase* op) {
  generated_op_ = op ? op->shared_from_this() : nullptr;
}

void VarBase::ClearGradient() {
  if (grad_var_ && !stop_gradient_) {
    auto* grad_t = grad_var_->var_.GetMutable<framework::LoDTensor>();
    if (grad_t->IsInitialized()) {
      auto* dev_ctx =
          platform::DeviceContextPool::Instance().Get(grad_t->place());
      operators::math::set_constant(*dev_ctx, grad_t, 0.0);
    }
  }
}

std::shared_ptr<VarBase> VarBase::NewVarBase(const platform::Place& dst_place,
                                             const bool blocking) const {
  PADDLE_ENFORCE(var_.IsInitialized() && var_.IsType<framework::LoDTensor>(),
                 "Variable must be initialized and type of LoDTensor when "
                 "getting numpy tensor");

  // TODO(minqiyang): change this after move unique_name generator to CXX
  auto& src_tensor = var_.Get<framework::LoDTensor>();

  auto new_var = std::make_shared<VarBase>(false);

  auto* dst_tensor = new_var->var_.GetMutable<framework::LoDTensor>();
  dst_tensor->set_lod(src_tensor.lod());

  framework::TensorCopy(src_tensor, dst_place, dst_tensor);
  if (blocking) {
    platform::DeviceContextPool::Instance().Get(dst_place)->Wait();
    auto src_place = src_tensor.place();
    if (!(src_place == dst_place)) {
      platform::DeviceContextPool::Instance().Get(src_place)->Wait();
    }
  }

  if (platform::is_gpu_place(dst_place)) {
    VLOG(3) << "copy tensor " << Name() << " from gpu";
  }

  return new_var;
}

OpBase::OpBase(Tracer* tracer, size_t id, const std::string& type,
               const NameVarBaseMap& ins, const NameVarBaseMap& outs,
               framework::AttributeMap attrs, const platform::Place& place,
               bool trace_backward)
    : tracer_(tracer), id_(id), type_(type), place_(place) {
  platform::RecordEvent event(type_);
  info_ = &(framework::OpInfoMap::Instance().Get(type_));

  // Step 1: Run forward
  if (info_->Checker() != nullptr) {
    info_->Checker()->Check(&attrs);
  }

  auto input_name_map = CreateVarNameMap<true>(info_, type_, ins);
  auto output_name_map = CreateVarNameMap<false>(info_, type_, outs);
  auto op = framework::OpRegistry::CreateOp(type_, input_name_map,
                                            output_name_map, attrs);
  RunOp(op, ins, outs);
  // Step 2: Trace backward if needed
  if (trace_backward) {
    framework::OpDesc fwd_op(type_, std::move(input_name_map),
                             std::move(output_name_map), std::move(attrs));
    TraceBackward(std::move(fwd_op), ins, outs);
  }
}

OpBase::OpBase(Tracer* tracer, size_t id, const framework::OpDesc& op_desc,
               const NameVarBaseMap& ins, const NameVarBaseMap& outs,
               const platform::Place& place, bool trace_backward)
    : tracer_(tracer), id_(id), type_(op_desc.Type()), place_(place) {
  platform::RecordEvent event(type_);
  info_ = &(framework::OpInfoMap::Instance().Get(type_));
  auto op = framework::OpRegistry::CreateOp(op_desc);

  RunOp(op, ins, outs);

  if (trace_backward) {
    TraceBackward(op_desc, ins, outs);
  }
}

void OpBase::RunOp(const std::unique_ptr<framework::OperatorBase>& op,
                   const NameVarBaseMap& ins, const NameVarBaseMap& outs) {
  auto* op_kernel = dynamic_cast<framework::OperatorWithKernel*>(op.get());
  PADDLE_ENFORCE_NOT_NULL(op_kernel, "only support op with kernel");
  VLOG(2) << "Create op " << type_ << " with input " << ins.size()
          << " and output " << outs.size();
  if (info_->infer_var_type_) {
    RuntimeInferVarTypeContext infer_var_type_ctx(ins, &outs,
                                                  op_kernel->Attrs());
    info_->infer_var_type_(&infer_var_type_ctx);
  }

  // Initialize output var type
  for (auto& var_pair : outs) {
    for (auto& var : var_pair.second) {
      InitializeVariable(var->MutableVar(), var->Type());
    }
  }

  VLOG(3) << "Running Op " << op->Type();
  auto runtime_ctx = PrepareRuntimeContext(ins, outs);
  auto runtime_place = PreparedOp::GetExpectedPlace(place(), ins);

  auto prepared_op =
      PreparedOp::Prepare(runtime_ctx, *op_kernel, runtime_place);

  prepared_op.Run();

  VLOG(3) << "Running Op " << op->Type() << " ends";
}

void OpBase::TraceBackward(const framework::OpDesc& fwd_op,
                           const NameVarBaseMap& ins,
                           const NameVarBaseMap& outs) {
  PADDLE_ENFORCE(grad_op_descs_.empty(),
                 "Grad op descs must be initialized here");
  PADDLE_ENFORCE(preceding_ops_.empty(),
                 "Preceding ops must be initialized here");

  // grad_to_var is a map of framework::GradVarName(in_var_name/out_var_name) ->
  // in_var_name/out_var_name
  std::unordered_map<std::string, std::string> grad_to_var;

  // NOTE(minqiyang): We don't support control flow op in imperative now
  // Add grad_block_ when we want to support it
  grad_op_descs_ = CreateGradOpDescs(fwd_op, {}, {}, &grad_to_var);

  size_t grad_op_num = grad_op_descs_.size();

  VLOG(5) << "Create " << grad_op_num << " grad op desc(s) to op " << type_;

  if (grad_op_num == 0) return;

  // Build a map to record var_name -> std::shared_ptr<VarBase>*,
  // so that we can find suitable var in grad op descs
  std::unordered_map<std::string, const std::shared_ptr<VarBase>*> name_to_var;
  for (auto& pair : ins) {
    for (auto& var : pair.second) {
      auto& var_ptr = name_to_var[var->Name()];
      PADDLE_ENFORCE(var_ptr == nullptr || var_ptr->get() == var.get(),
                     "There are different variables with same name %s",
                     var->Name());
      var_ptr = &var;
    }
  }

  for (auto& pair : outs) {
    for (auto& var : pair.second) {
      auto& var_ptr = name_to_var[var->Name()];
      PADDLE_ENFORCE(var_ptr == nullptr || var_ptr->get() == var.get(),
                     "There are different variables with same name %s",
                     var->Name());
      var_ptr = &var;
    }
  }

  // Build backward ins and outs
  bwd_ins_.resize(grad_op_num);
  bwd_outs_.resize(grad_op_num);

  for (size_t i = 0; i < grad_op_num; ++i) {
    for (auto& grad_ins : grad_op_descs_[i]->Inputs()) {
      if (grad_ins.second.empty()) continue;
      auto& bwd_in = bwd_ins_[i][grad_ins.first];
      bwd_in.reserve(grad_ins.second.size());

      for (auto& grad_in_var_name : grad_ins.second) {
        auto iter = grad_to_var.find(grad_in_var_name);

        if (iter != grad_to_var.end()) {
          // If it is a grad var, find its coresponding forward var
          auto& fwd_var_name = iter->second;
          auto fwd_var_iter = name_to_var.find(fwd_var_name);
          PADDLE_ENFORCE(fwd_var_iter != name_to_var.end(),
                         "Cannot find forward variable named %s", fwd_var_name);

          bwd_in.emplace_back((*(fwd_var_iter->second))->GradVarBase());
        } else {
          // If it is a forward var, just add it
          auto fwd_var_iter = name_to_var.find(grad_in_var_name);
          PADDLE_ENFORCE(fwd_var_iter != name_to_var.end(),
                         "Cannot find forward variable named %s",
                         grad_in_var_name);
          bwd_in.emplace_back(*(fwd_var_iter->second));
        }

        VLOG(2) << "Set backward input " << grad_ins.first << " of " << type_
                << " to be " << bwd_in.back()->Name();
      }
    }

    for (auto& grad_outs : grad_op_descs_[i]->Outputs()) {
      if (grad_outs.second.empty()) continue;
      auto& bwd_out = bwd_outs_[i][grad_outs.first];
      bwd_out.reserve(grad_outs.second.size());

      for (auto& grad_out_var_name : grad_outs.second) {
        auto iter = grad_to_var.find(grad_out_var_name);
        PADDLE_ENFORCE(iter != grad_to_var.end(),
                       "Cannot find output of input grad %s in op %s",
                       grad_out_var_name, type_);
        auto fwd_var_iter = name_to_var.find(iter->second);
        PADDLE_ENFORCE(fwd_var_iter != name_to_var.end(),
                       "Cannot find forward variable named %s", iter->second);
        bwd_out.emplace_back((*(fwd_var_iter->second))->GradVarBase());
        VLOG(2) << "Set backward output " << grad_outs.first << " of " << type_
                << " to be "
                << (bwd_out.back() ? bwd_out.back()->Name() : "nullptr");

        auto* preceding_op = (*(fwd_var_iter->second))->GeneratedOp();

        if (preceding_op) {
          preceding_ops_.insert(preceding_op);
        }
      }
    }
  }
}

void OpBase::ClearBackwardTrace() {
  grad_op_descs_.clear();
  preceding_ops_.clear();
  bwd_ins_.clear();
  bwd_outs_.clear();
}

/* Autograd implementation */

struct GradAccumulator {
  explicit GradAccumulator(VarBase* var,
                           const detail::BackwardStrategy& backward_strategy)
      : var_(var), backward_strategy_(backward_strategy) {}

  void Add(std::shared_ptr<VarBase> var, size_t trace_id) {
    if (!backward_strategy_.sorted_sum_gradient_) {
      if (var_->MutableVar()->IsInitialized()) {
        TensorAdd(var->Var(), var_->MutableVar());
      } else {
        *(var_->MutableVar()) = std::move(*(var->MutableVar()));
      }
      return;
    }

    if (ref_cnt_ == 1) {
      *(var_->MutableVar()) = std::move(*(var->MutableVar()));
    } else {
      if (tmp_grad_vars_.empty()) {
        tmp_grad_vars_.reserve(ref_cnt_);
      }

      tmp_grad_vars_.emplace_back(std::move(var), trace_id);
      if (tmp_grad_vars_.size() != ref_cnt_) {
        return;
      }

      std::sort(tmp_grad_vars_.begin(), tmp_grad_vars_.end(),
                [](const std::pair<std::shared_ptr<VarBase>, size_t>& p1,
                   const std::pair<std::shared_ptr<VarBase>, size_t>& p2) {
                  return p1.second > p2.second;
                });

      *(var_->MutableVar()) =
          std::move(*(tmp_grad_vars_[0].first->MutableVar()));
      for (size_t i = 1; i < tmp_grad_vars_.size(); ++i) {
        TensorAdd(tmp_grad_vars_[i].first->Var(), var_->MutableVar());
      }

      tmp_grad_vars_.clear();
    }
  }

  size_t IncreaseRefCnt() { return ++ref_cnt_; }

 private:
  VarBase* var_;
  detail::BackwardStrategy backward_strategy_;
  size_t ref_cnt_{0};
  std::vector<std::pair<std::shared_ptr<VarBase>, size_t>> tmp_grad_vars_;
  std::vector<size_t> trace_ids_;
};

class AutoGradImpl {
 public:
  explicit AutoGradImpl(OpBase* op, const detail::BackwardStrategy& strategy);

  void operator()();

 private:
  void PrepareDeps();

  bool CheckBackwardInputs(OpBase* op);

  void CheckBackwardOutputs(OpBase* op);

  void SumGradient(OpBase* op, std::shared_ptr<VarBase> src, VarBase* dst) {
    auto iter = accumulators_.find(dst);
    PADDLE_ENFORCE(iter != accumulators_.end(),
                   "Cannot find gradient of variable %s", dst->Name());
    iter->second.Add(std::move(src), op->id());
  }

  Tracer* tracer_;
  OpBase* op_;
  detail::BackwardStrategy backward_strategy_;
  std::unordered_map<OpBase*, size_t> op_deps_;
  std::unordered_map<VarBase*, GradAccumulator> accumulators_;
};

void AutoGrad(VarBase* var, const detail::BackwardStrategy& strategy) {
  auto* op = var->GeneratedOp();
  if (!op) {
    VLOG(3) << "Skip auto grad since generated op is nullptr";
    return;
  }

  platform::RecordEvent record_event("Imperative Backward");
  VLOG(3) << "start backward";

  PADDLE_ENFORCE(var->HasGradVar(), "Grad variable not exist for variable %s",
                 var->Name());

  auto& fwd_var = var->Var().Get<framework::LoDTensor>();
  auto* grad_var =
      var->GradVarBase()->MutableVar()->GetMutable<framework::LoDTensor>();
  auto* dev_ctx = platform::DeviceContextPool::Instance().Get(fwd_var.place());
  grad_var->Resize(fwd_var.dims());
  grad_var->mutable_data(fwd_var.place(), fwd_var.type());
  operators::math::set_constant(*dev_ctx, grad_var, 1.0);

  AutoGradImpl functor(op, strategy);
  functor();
}

AutoGradImpl::AutoGradImpl(OpBase* op, const detail::BackwardStrategy& strategy)
    : tracer_(op->tracer()), op_(op), backward_strategy_(strategy) {
  // Step 1: build a graph to record op dependency
  PrepareDeps();
}

bool AutoGradImpl::CheckBackwardInputs(OpBase* op) {
  for (auto& in : op->BackwardInputs()) {
    for (auto& pair : in) {
      for (auto& var : pair.second) {
        if (var && !var->StopGradient()) {
          return true;
        }
      }
    }
  }
  return false;
}

void AutoGradImpl::CheckBackwardOutputs(OpBase* op) {
  for (auto& out : op->BackwardOutputs()) {
    for (auto& pair : out) {
      for (auto& var : pair.second) {
        if (!var) continue;

        auto pair = accumulators_.emplace(
            var.get(), GradAccumulator(var.get(), backward_strategy_));
        auto cnt = pair.first->second.IncreaseRefCnt();
        VLOG(2) << "Prepare to acccumulate variable grad " << var->Name()
                << "with reference count " << cnt;
      }
    }
  }
}

void AutoGradImpl::PrepareDeps() {
  PADDLE_ENFORCE(op_deps_.empty(), "Op deps must be initialized here");
  PADDLE_ENFORCE(accumulators_.empty(),
                 "Accumulators must be initialized here");

  std::queue<OpBase*> q;
  std::unordered_set<OpBase*> visited;
  q.push(op_);
  visited.insert(op_);

  while (!q.empty()) {
    auto* cur_op = q.front();
    q.pop();
    VLOG(2) << "Checking grads of op " << cur_op->Type();

    if (!CheckBackwardInputs(cur_op)) {
      // TODO(zjl): clear ops that do not need grad before running autograd
      VLOG(2) << "Stop checking preceding ops of " << cur_op->Type()
              << " because all of its backward inputs is stop_gradient=True";
      continue;
    }

    CheckBackwardOutputs(cur_op);

    auto& preceding_ops = cur_op->PrecedingOps();
    for (auto* preceding_op : preceding_ops) {
      PADDLE_ENFORCE_NOT_NULL(preceding_op);
      ++op_deps_[preceding_op];
      if (visited.count(preceding_op) == 0) {
        visited.insert(preceding_op);
        q.push(preceding_op);
      }
    }
  }
}

void AutoGradImpl::operator()() {
  std::queue<OpBase*> q;
  q.push(op_);

  while (!q.empty()) {
    OpBase* cur_op = q.front();
    q.pop();

    // Step 1: Run Backward
    auto& grad_descs = cur_op->GradOpDescs();
    auto& bwd_ins = cur_op->BackwardInputs();
    auto& bwd_outs = cur_op->BackwardOutputs();
    size_t grad_op_num = grad_descs.size();

    PADDLE_ENFORCE_EQ(grad_op_num, bwd_ins.size());
    PADDLE_ENFORCE_EQ(grad_op_num, bwd_outs.size());

    for (size_t i = 0; i < grad_op_num; ++i) {
      NameVarBaseMap tmp_outs;
      // A var may be coresponding to several grad var in one op
      std::unordered_map<VarBase*, std::vector<std::shared_ptr<VarBase>>>
          var_map;
      size_t counter = 0;
      for (auto& bwd_out : bwd_outs[i]) {
        auto& tmp_var_list = tmp_outs[bwd_out.first];
        tmp_var_list.reserve(bwd_out.second.size());
        for (auto& var : bwd_out.second) {
          auto tmp_var = std::make_shared<VarBase>(false);  // Do not need grad
          tmp_var->SetName("Gtmp@" + std::to_string(counter++));
          tmp_var_list.emplace_back(tmp_var);
          if (var) {
            var_map[var.get()].emplace_back(std::move(tmp_var));
          }
        }
      }

      VLOG(2) << "Trace grad op " << grad_descs[i]->Type() << " starts";
      tracer_->TraceOp(*(grad_descs[i]), bwd_ins[i], tmp_outs, cur_op->place(),
                       false);
      VLOG(2) << "Trace grad op " << grad_descs[i]->Type() << " ends";

      // Step 2: Sum Gradient
      for (auto& var_pair : var_map) {
        auto* dst_var = var_pair.first;
        if (dst_var == nullptr) continue;
        for (auto& src_var : var_pair.second) {
          VLOG(2) << "Sum gradient of variable " << dst_var->Name()
                  << " after op " << grad_descs[i]->Type();
          SumGradient(cur_op, src_var, dst_var);
          VLOG(2) << "Sum gradient ends of variable " << dst_var->Name()
                  << " after op " << grad_descs[i]->Type();
        }
      }
    }

    VLOG(2) << "Get preceding op number";
    size_t num = cur_op->PrecedingOps().size();
    VLOG(2) << "Preceding op number is " << num;

    // Step 3: Collect ready ops
    for (auto* preceding_op : cur_op->PrecedingOps()) {
      PADDLE_ENFORCE_NOT_NULL(preceding_op);
      auto iter = op_deps_.find(preceding_op);
      VLOG(2) << "Find preceding op of " << cur_op->Type();
      PADDLE_ENFORCE(iter != op_deps_.end(), "Cannot find op %s",
                     cur_op->Type());
      VLOG(2) << "Found preceding op of " << cur_op->Type();
      if (--(iter->second) == 0) {
        q.push(preceding_op);
        VLOG(2) << "Push preceding op " << preceding_op->Type()
                << " into queue";
      }
    }

    // Step 4: Delete op to collect unused variables
    VLOG(2) << "Remove op after op " << cur_op->Type() << " runs";
    tracer_->RemoveOp(cur_op);
  }

  VLOG(2) << "Clear left op in tracer";
  tracer_->Clear();
}

}  // namespace imperative
}  // namespace paddle
