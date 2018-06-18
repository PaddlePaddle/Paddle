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
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>

#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(benchmark);
DEFINE_bool(check_nan_inf, false,
            "Checking whether operator produce NAN/INF or not. It will be "
            "extremely slow so please use this flag wisely.");

namespace paddle {
namespace framework {

std::vector<std::tuple<platform::Place, LibraryType>> kKernelPriority = {
    std::make_tuple(platform::CUDAPlace(0), LibraryType::kCUDNN),
    std::make_tuple(platform::CUDAPlace(0), LibraryType::kPlain),
    std::make_tuple(platform::CPUPlace(), LibraryType::kMKLDNN),
    std::make_tuple(platform::CPUPlace(), LibraryType::kPlain),
};

proto::VarType::Type GetDataTypeOfVar(const Variable* var) {
  if (var->IsType<framework::LoDTensor>()) {
    return framework::ToDataType(var->Get<framework::LoDTensor>().type());
  } else if (var->IsType<framework::SelectedRows>()) {
    return framework::ToDataType(
        var->Get<framework::SelectedRows>().value().type());
  } else {
    PADDLE_THROW("Var should be LoDTensor or SelectedRows");
  }
}

static DDim GetDims(const Scope& scope, const std::string& name,
                    bool get_actual_dim = false) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return DDim({-1});
  }

  if (var->IsType<LoDTensor>()) {
    return var->Get<LoDTensor>().dims();
  } else if (var->IsType<SelectedRows>()) {
    if (get_actual_dim) {
      return var->Get<SelectedRows>().value().dims();
    } else {
      return var->Get<SelectedRows>().GetCompleteDims();
    }
  } else {
    return DDim({-1});
  }
}

static int GetRowSize(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  if (var == nullptr) {
    return -1;
  }

  if (var->IsType<SelectedRows>()) {
    return var->Get<SelectedRows>().rows().size();
  }

  return -1;
}

static LoD GetLoD(const Scope& scope, const std::string& name) {
  Variable* var = scope.FindVar(name);
  auto default_lod = LoD({{}});

  if (var == nullptr) {
    return default_lod;
  }

  if (var->IsType<LoDTensor>()) {
    return var->Get<LoDTensor>().lod();
  } else {
    return default_lod;
  }
}

void OperatorBase::Run(const Scope& scope, const platform::Place& place) {
  VLOG(10) << "- " << DebugStringEx(&scope);
  if (platform::is_gpu_place(place)) {
#ifndef PADDLE_WITH_CUDA
    PADDLE_THROW("Cannot run operator on place %s", place);
#else
    auto dev_id = boost::get<platform::CUDAPlace>(place).device;
    platform::SetDeviceId(dev_id);
#endif
  }
  RunImpl(scope, place);
  VLOG(10) << "+ " << DebugStringEx(&scope);
}

bool OperatorBase::HasInputs(const std::string& name) const {
  if (inputs_.find(name) != inputs_.end()) {
    return true;
  } else {
    return false;
  }
}

std::string OperatorBase::Input(const std::string& name) const {
  auto& ins = Inputs(name);
  PADDLE_ENFORCE_LE(ins.size(), 1UL,
                    "Operator %s's input %s should contain only one variable.",
                    type_, name);
  return ins.empty() ? kEmptyVarName : ins[0];
}

const std::vector<std::string>& OperatorBase::Inputs(
    const std::string& name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE(it != inputs_.end(), "Operator %s does not have the input %s.",
                 type_, name);
  return it->second;
}

bool OperatorBase::HasOutputs(const std::string& name) const {
  if (outputs_.find(name) != outputs_.end()) {
    return true;
  } else {
    return false;
  }
}

std::string OperatorBase::Output(const std::string& name) const {
  auto& outs = Outputs(name);
  PADDLE_ENFORCE_LE(outs.size(), 1UL,
                    "Operator %s's output %s should contain only one variable.",
                    type_, name);
  return outs.empty() ? kEmptyVarName : outs[0];
}

const std::vector<std::string>& OperatorBase::Outputs(
    const std::string& name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE(it != outputs_.end(),
                 "Operator %s does not have an output called %s.", type_, name);
  return it->second;
}

std::string OperatorBase::DebugStringEx(const Scope* scope) const {
  std::stringstream ss;
  ss << "Op(" << type_ << "), inputs:{";
  for (auto it = inputs_.begin(); it != inputs_.end();) {
    auto& input = *it;
    ss << input.first << "[";
    for (size_t i = 0; i < input.second.size(); ++i) {
      ss << input.second[i];
      if (scope) {
        int row_size = GetRowSize(*scope, input.second[i]);
        if (row_size >= 0) {
          ss << "[row_size=" << row_size << "]";
        }
        ss << "[" << GetDims(*scope, input.second[i], true) << "]";
        ss << "(" << GetLoD(*scope, input.second[i]) << ")";
      }
      if (i != input.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != inputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}, outputs:{";
  for (auto it = outputs_.begin(); it != outputs_.end();) {
    auto& output = *it;
    ss << output.first << "[";
    for (size_t i = 0; i < output.second.size(); ++i) {
      ss << output.second[i];
      if (scope) {
        int row_size = GetRowSize(*scope, output.second[i]);
        if (row_size >= 0) {
          ss << "[row_size=" << row_size << "]";
        }
        ss << "[" << GetDims(*scope, output.second[i], true) << "]";
        ss << "(" << GetLoD(*scope, output.second[i]) << ")";
      }
      if (i != output.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != outputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}.";
  return ss.str();
}

OperatorBase::OperatorBase(const std::string& type,
                           const VariableNameMap& inputs,
                           const VariableNameMap& outputs,
                           const AttributeMap& attrs)
    : type_(type), inputs_(inputs), outputs_(outputs), attrs_(attrs) {
  GenerateTemporaryNames();
  CheckAllInputOutputSet();
}

std::vector<std::string> OperatorBase::InputVars() const {
  std::vector<std::string> ret_val;
  for (auto& o : inputs_) {
    ret_val.reserve(ret_val.size() + o.second.size());
    ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
  }
  return ret_val;
}

std::vector<std::string> OperatorBase::OutputVars(bool has_intermediate) const {
  std::vector<std::string> ret_val;
  if (has_intermediate) {
    // push all outputs into ret_val
    for (auto& o : outputs_) {
      ret_val.reserve(ret_val.size() + o.second.size());
      ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
    }
    return ret_val;
  }
  auto& info = OpInfoMap::Instance().Get(Type());

  // get all OpProto::Var for outputs
  for (auto& o : info.Proto().outputs()) {
    // ignore all intermediate output
    if (o.intermediate()) continue;
    auto out = outputs_.find(o.name());
    if (out != outputs_.end()) {
      ret_val.reserve(ret_val.size() + out->second.size());
      ret_val.insert(ret_val.end(), out->second.begin(), out->second.end());
    }
  }
  return ret_val;
}

void OperatorBase::CheckAllInputOutputSet() const {
  auto& info_map = OpInfoMap::Instance();
  auto* op_info = info_map.GetNullable(Type());
  if (op_info == nullptr || op_info->proto_ == nullptr) return;

  for (auto& in : op_info->Proto().inputs()) {
    if (!in.dispensable()) {
      PADDLE_ENFORCE(inputs_.find(in.name()) != inputs_.end(),
                     "Operator %s's input, %s, is not set", Type(), in.name());
    }
  }

  for (auto& out : op_info->Proto().outputs()) {
    if (!out.dispensable()) {
      PADDLE_ENFORCE(outputs_.find(out.name()) != outputs_.end(),
                     "Operator %s's output, %s, is not set", Type(),
                     out.name());
    }
  }
}

void OperatorBase::GenerateTemporaryNames() {
  static std::atomic<size_t> gUniqId(0UL);
  for (auto& output : outputs_) {
    for (auto& output_name : output.second) {
      if (output_name == kTempVarName) {
        output_name += type_;
        output_name += "@";
        output_name += std::to_string(gUniqId.fetch_add(1));
      }
    }
  }
}

static bool VarIsTensor(const Variable* var) {
  return var->IsType<LoDTensor>() || var->IsType<SelectedRows>();
}

static const Tensor* GetTensorFromVar(Variable* var) {
  if (var->IsType<LoDTensor>()) {
    return var->GetMutable<LoDTensor>();
  } else if (var->IsType<SelectedRows>()) {
    return var->GetMutable<SelectedRows>()->mutable_value();
  } else {
    PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                 var->Type().name());
  }
}

static Tensor* GetMutableTensorFromVar(Variable* var) {
  if (var->IsType<LoDTensor>()) {
    return var->GetMutable<LoDTensor>();
  } else if (var->IsType<SelectedRows>()) {
    return var->GetMutable<SelectedRows>()->mutable_value();
  } else {
    PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                 var->Type().name());
  }
}

bool ExecutionContext::HasInput(const std::string& name) const {
  if (!op_.HasInputs(name)) {
    return false;
  }
  auto& ins = Inputs(name);
  size_t length = ins.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(length, 1UL,
                    "Input %s should not have more than one inputs", name);
  auto arg = ins[0];
  auto* var = arg == kEmptyVarName ? nullptr : scope_.FindVar(arg);
  return var != nullptr;
}

bool ExecutionContext::HasOutput(const std::string& name) const {
  if (!op_.HasOutputs(name)) {
    return false;
  }
  auto& outs = Outputs(name);
  size_t length = outs.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(length, 1UL,
                    "Output %s should not have more than one inputs", name);
  auto arg = outs[0];
  auto* var = arg == kEmptyVarName ? nullptr : scope_.FindVar(arg);
  return var != nullptr;
}

template <>
const Tensor* ExecutionContext::Input<Tensor>(const std::string& name) const {
  auto* var = InputVar(name);
  return var == nullptr ? nullptr
                        : GetTensorFromVar(const_cast<Variable*>(var));
}

template <>
const std::vector<const Tensor*> ExecutionContext::MultiInput<Tensor>(
    const std::string& name) const {
  auto names = op().Inputs(name);
  std::vector<const Tensor*> res;
  res.reserve(names.size());
  std::transform(names.begin(), names.end(), std::back_inserter(res),
                 [&](const std::string& sub_name) {
                   auto var = scope_.FindVar(sub_name);
                   return var == nullptr ? nullptr : GetTensorFromVar(var);
                 });
  return res;
}

template <>
Tensor* ExecutionContext::Output<Tensor>(const std::string& name) const {
  auto var = OutputVar(name);
  return var == nullptr ? nullptr : GetMutableTensorFromVar(var);
}

template <>
std::vector<Tensor*> ExecutionContext::MultiOutput<Tensor>(
    const std::string& name) const {
  auto names = op().Outputs(name);
  std::vector<Tensor*> res;
  res.reserve(names.size());
  std::transform(names.begin(), names.end(), std::back_inserter(res),
                 [&](const std::string& sub_name) {
                   auto var = scope_.FindVar(sub_name);
                   return var == nullptr ? nullptr
                                         : GetMutableTensorFromVar(var);
                 });
  return res;
}

bool OpSupportGPU(const std::string& op_type) {
  auto& all_kernels = OperatorWithKernel::AllOpKernels();
  auto it = all_kernels.find(op_type);
  if (it == all_kernels.end()) {
    // All control operator must support GPU
    return true;
  }
  for (auto& kern_pair : it->second) {
    if (platform::is_gpu_place(kern_pair.first.place_)) {
      return true;
    }
  }
  return false;
}

class RuntimeInferShapeContext : public InferShapeContext {
 public:
  RuntimeInferShapeContext(const OperatorBase& op, const Scope& scope)
      : op_(op), scope_(scope) {}

  bool HasInput(const std::string& name) const override {
    if (!op_.HasInputs(name)) {
      return false;
    }
    auto& ins = Inputs(name);
    size_t length = ins.size();
    if (length == 0) {
      return false;
    }
    PADDLE_ENFORCE_EQ(length, 1UL,
                      "Input %s should not have more than one inputs", name);
    auto ipt = ins[0];
    auto* var = ipt == kEmptyVarName ? nullptr : scope_.FindVar(ipt);
    return var != nullptr;
  }

  bool HasOutput(const std::string& name) const override {
    if (!op_.HasOutputs(name)) {
      return false;
    }
    auto& outs = Outputs(name);
    size_t length = outs.size();
    if (length == 0) {
      return false;
    }
    PADDLE_ENFORCE_EQ(length, 1UL,
                      "Output %s should not have more than one inputs", name);
    auto ipt = outs[0];
    auto* var = ipt == kEmptyVarName ? nullptr : scope_.FindVar(ipt);
    return var != nullptr;
  }

  bool HasInputs(const std::string& name) const override {
    if (!op_.HasInputs(name)) {
      return false;
    }
    auto inputs = op_.Inputs(name);
    if (inputs.empty()) {
      return false;
    }
    for (auto& input : inputs) {
      if (scope_.FindVar(input) == nullptr) {
        return false;
      }
    }
    return true;
  }

  bool HasOutputs(const std::string& name) const override {
    if (!op_.HasOutputs(name)) {
      return false;
    }
    auto outputs = op_.Outputs(name);
    if (outputs.empty()) {
      return false;
    }
    for (auto& output : outputs) {
      if (scope_.FindVar(output) == nullptr) {
        return false;
      }
    }
    return true;
  }

  AttrReader Attrs() const override { return AttrReader(op_.Attrs()); }

  const std::vector<std::string>& Inputs(
      const std::string& name) const override {
    return op_.Inputs(name);
  }

  const std::vector<std::string>& Outputs(
      const std::string& name) const override {
    return op_.Outputs(name);
  }

  void ShareLoD(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) const override {
    PADDLE_ENFORCE_LT(i, Inputs(in).size());
    PADDLE_ENFORCE_LT(j, Outputs(out).size());
    Variable* in_var = scope_.FindVar(Inputs(in)[i]);
    Variable* out_var = scope_.FindVar(Outputs(out)[j]);
    if (!in_var->IsType<LoDTensor>()) return;
    PADDLE_ENFORCE(out_var->IsType<LoDTensor>(),
                   "The %d-th output of Output(%s) must be LoDTensor.", j, out);
    auto in_tensor = in_var->Get<LoDTensor>();
    auto* out_tensor = out_var->GetMutable<LoDTensor>();
    out_tensor->set_lod(in_tensor.lod());

// TODO(dzhwinter) : reuse ShareLoD in most operators.
// Need to call ShareLayout explicitly in sequence related ops.
// Shall we have a better method to shared info between in/out Tensor?
#ifdef PADDLE_WITH_MKLDNN
    // Fix me: ugly workaround below
    // Correct solution:
    //    set_layout() should NOT be called here (i.e. ShareLoD). Instead,
    //    layout of output tensor should be set "manually" in Compute()
    //    of each OPKernel. The reason layout should NOT be shared between
    //    input and output "automatically" (now by InferShape()->ShareLoD())
    //    is that layout transform may occur after InferShape().
    // Workaround:
    //    Skip set_layout() when input layout is kMKLDNN
    //    This is to avoid kMKLDNN is populated wrongly into a non-MKLDNN
    //    OPKernel. In all MKLDNN OPkernel, set_layout(kMKLDNN) should be called
    //    in Compute()
    if (in_tensor.layout() != DataLayout::kMKLDNN)
#endif
      out_tensor->set_layout(in_tensor.layout());
  }

  void ShareLayout(const std::string& in, const std::string& out, size_t i = 0,
                   size_t j = 0) const {
    PADDLE_ENFORCE_LT(i, Inputs(in).size());
    PADDLE_ENFORCE_LT(j, Outputs(out).size());
    Variable* in_var = scope_.FindVar(Inputs(in)[i]);
    Variable* out_var = scope_.FindVar(Outputs(out)[j]);
    if (!in_var->IsType<LoDTensor>()) return;
    PADDLE_ENFORCE(out_var->IsType<LoDTensor>(),
                   "The %d-th output of Output(%s) must be LoDTensor.", j, out);
    auto in_tensor = in_var->Get<LoDTensor>();
    auto* out_tensor = out_var->GetMutable<LoDTensor>();
    out_tensor->set_layout(in_tensor.layout());
  }

  bool IsRuntime() const override { return true; }

 protected:
  DDim GetDim(const std::string& name) const override {
    Variable* var = scope_.FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(var);
    if (var->IsType<LoDTensor>()) {
      return var->Get<LoDTensor>().dims();
    } else if (var->IsType<SelectedRows>()) {
      return var->Get<SelectedRows>().GetCompleteDims();
    } else {
      PADDLE_THROW(
          "Only LoDTensor/SelectedRows support 'GetDim', but Variable %s's "
          "type_id is %s.",
          name, var->Type().name());
    }
  }

  std::vector<DDim> GetRepeatedDims(const std::string& name) const override {
    PADDLE_THROW("Only compile time support this method");
  }

  void SetDim(const std::string& name, const DDim& dim) override {
    Variable* var = scope_.FindVar(name);
    if (var->IsType<LoDTensor>()) {
      var->GetMutable<LoDTensor>()->Resize(dim);
    } else if (var->IsType<SelectedRows>()) {
      var->GetMutable<SelectedRows>()->set_height(dim[0]);
    } else {
      PADDLE_THROW("Variable %s type_id %s, expect LoDTensor/SelectedRows.",
                   name, var->Type().name());
    }
  }

  void SetRepeatedDims(const std::string& name,
                       const std::vector<DDim>& dims) override {
    PADDLE_THROW("Only compile time support this method");
  }

  proto::VarType::Type GetVarType(const std::string& name) const override {
    auto* var = scope_.FindVar(name);
    return ToVarType(var->Type());
  }

  InferShapeVarPtr GetVarPtr(const std::string& name) override {
    return scope_.FindVar(name);
  }

 private:
  const OperatorBase& op_;
  const Scope& scope_;
};

static void CheckTensorNANOrInf(const std::string& name,
                                const framework::Tensor& tensor) {
  if (tensor.memory_size() == 0) {
    return;
  }
  if (tensor.type().hash_code() != typeid(float).hash_code() &&   // NOLINT
      tensor.type().hash_code() != typeid(double).hash_code()) {  // NOLINT
    return;
  }
  PADDLE_ENFORCE(!framework::TensorContainsInf(tensor),
                 "Tensor %s contains Inf", name);
  PADDLE_ENFORCE(!framework::TensorContainsNAN(tensor),
                 "Tensor %s contains NAN", name);
}

void OperatorWithKernel::RunImpl(const Scope& scope,
                                 const platform::Place& place) const {
  RuntimeInferShapeContext infer_shape_ctx(*this, scope);
  this->InferShape(&infer_shape_ctx);
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);

  // For profiling, don't move out of this function because that will result
  // in the failure of multi-GPU profiling.
  platform::RecordEvent record_event(Type(), dev_ctx);
  // check if op[type] has kernel registered.
  auto& all_op_kernels = AllOpKernels();
  auto kernels_iter = all_op_kernels.find(type_);
  if (kernels_iter == all_op_kernels.end()) {
    PADDLE_THROW(
        "There are no kernels which are registered in the %s operator.", type_);
  }

  ExecutionContext ctx(*this, scope, *dev_ctx);

  OpKernelMap& kernels = kernels_iter->second;

  // TODO(dzhwinter) : kernel fallback mechanism will be added when all the
  // transform functions are ready.

  // for (auto& candidate : kKernelPriority) {
  //   Do selection
  // }

  auto expected_kernel_key = this->GetExpectedKernelType(ctx);
  VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

  auto kernel_iter = kernels.find(expected_kernel_key);
  if (kernel_iter == kernels.end()) {
    PADDLE_THROW("op %s does not have kernel for %s", type_,
                 KernelTypeToString(expected_kernel_key));
  }

  // do data transform
  Scope& new_scope = scope.NewScope();

  std::vector<std::string> inplace_vars;
  for (auto& var_name_item : this->Inputs()) {
    for (auto& var_name : var_name_item.second) {
      auto* var = scope.FindVar(var_name);
      if (var && VarIsTensor(var)) {
        auto* tensor_in = GetTensorFromVar(var);
        if (tensor_in->IsInitialized()) {
          auto kernel_type_for_var = this->GetKernelTypeForVar(
              var_name_item.first, *tensor_in, expected_kernel_key);
          if (TransFromNeeded(kernel_type_for_var, expected_kernel_key)) {
            auto out_var_names = OutputVars(true);
            if (std::find(out_var_names.begin(), out_var_names.end(),
                          var_name) != out_var_names.end()) {
              inplace_vars.push_back(var_name);
            }
            VLOG(3) << "Transform Variable " << var_name << " from "
                    << kernel_type_for_var << " to " << expected_kernel_key;
            auto* trans_var = new_scope.Var(var_name);
            std::shared_ptr<Tensor> out(new Tensor);
            DataTransform(expected_kernel_key, kernel_type_for_var, *tensor_in,
                          out.get());
            CopyVariableWithTensor(*var, *(out.get()), trans_var);
          }
        }
      }
    }
  }

  auto* new_dev_ctx = pool.Get(expected_kernel_key.place_);
  kernel_iter->second->Compute(
      ExecutionContext(*this, new_scope, *new_dev_ctx));

  for (auto& var_name : inplace_vars) {
    VLOG(3) << "share inplace var " + var_name + " back to it's original scope";
    auto* original_tensor = GetMutableTensorFromVar(scope.FindVar(var_name));
    auto* transformed_tensor = GetTensorFromVar(new_scope.FindVar(var_name));
    original_tensor->ShareDataWith(*transformed_tensor);
  }

  /*For profiling/benchmark only*/
  if (FLAGS_benchmark) {
    new_dev_ctx->Wait();
  }

  if (FLAGS_check_nan_inf) {
    for (auto& vname : OutputVars(true)) {
      auto* var = new_scope.FindVar(vname);
      if (var == nullptr) continue;
      if (var->IsType<framework::LoDTensor>()) {
        CheckTensorNANOrInf(vname, var->Get<framework::LoDTensor>());
      }
    }
  }
}

proto::VarType::Type OperatorWithKernel::IndicateDataType(
    const ExecutionContext& ctx) const {
  auto& scope = ctx.scope();
  int data_type = -1;
  for (auto& input : this->inputs_) {
    for (auto& ipt_name : input.second) {
      auto* var = scope.FindVar(ipt_name);
      if (var != nullptr) {
        const Tensor* t = nullptr;
        if (var->IsType<Tensor>()) {
          t = &var->Get<Tensor>();
        } else if (var->IsType<LoDTensor>()) {
          t = &var->Get<LoDTensor>();
        } else if (var->IsType<SelectedRows>()) {
          t = &(var->Get<SelectedRows>().value());
        }
        if (t != nullptr) {
          int tmp = static_cast<int>(ToDataType(t->type()));
          PADDLE_ENFORCE(
              tmp == data_type || data_type == -1,
              "DataType of Paddle Op %s must be the same. Get %d != %d", Type(),
              data_type, tmp);
          data_type = tmp;
        }
      }
    }
  }
  PADDLE_ENFORCE(data_type != -1, "DataType should be indicated by input");
  return static_cast<proto::VarType::Type>(data_type);
}

OpKernelType OperatorWithKernel::GetExpectedKernelType(
    const ExecutionContext& ctx) const {
  return OpKernelType(IndicateDataType(ctx), ctx.GetPlace());
}

OpKernelType OperatorWithKernel::GetKernelTypeForVar(
    const std::string& var_name, const Tensor& tensor,
    const OpKernelType& expected_kernel_type) const {
  return OpKernelType(expected_kernel_type.data_type_, tensor.place(),
                      tensor.layout());
}

}  // namespace framework
}  // namespace paddle
